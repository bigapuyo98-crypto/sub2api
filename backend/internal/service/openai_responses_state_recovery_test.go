package service

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const crossResourceStateError = `{"error":{"code":null,"type":"invalid_request_error","message":"The requested item was created under a different Azure OpenAI resource. Use the same resource that created the item to access it."}}`
const maskedResourceStateError = `{"error":{"code":null,"type":"invalid_request_error","param":"","message":"OpenAI Responses bad request: The requested item was created under a different *** OpenAI resource. Use the same resource that created the item to access it. [trace_id=1570ba9a03970045783f90d21e7ef16f]"}}`
const encryptedReplayStateError = `{"error":{"code":null,"type":"invalid_request_error","message":"The encrypted content gAAA_fixture could not be verified. Reason: Encrypted content could not be decrypted or parsed."}}`
const stateRecoveryInput = `{"model":"gpt-5","stream":false,"store":true,"input":[{"type":"reasoning","id":"rs_history","encrypted_content":"fixture-cipher","summary":[{"type":"summary_text","text":"keep reasoning summary"}]},{"type":"message","id":"msg_history","role":"assistant","content":[{"type":"output_text","text":"keep answer","nonce":9007199254740993}]},{"type":"function_call","id":"fc_history","call_id":"call_shared","name":"lookup","arguments":"{\"n\":9007199254740993}"},{"type":"function_call_output","call_id":"call_shared","output":"keep tool result"},{"role":"user","content":"next question"}],"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`

func stateRecoveryResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func stateRecoverySuccess() *http.Response {
	return stateRecoveryResponse(http.StatusOK, `{"id":"resp_recovered","model":"gpt-5","output":[],"usage":{"input_tokens":7,"output_tokens":3}}`)
}

func newStateRecoveryForward(t *testing.T, passthrough bool, responses ...*http.Response) (*OpenAIGatewayService, *gin.Context, *Account, *httpUpstreamRecorder, *httptest.ResponseRecorder) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	upstream := &httpUpstreamRecorder{responses: responses}
	cfg := &config.Config{}
	cfg.Security.URLAllowlist.Enabled = false
	svc := &OpenAIGatewayService{cfg: cfg, httpUpstream: upstream}
	account := &Account{
		ID: 73, Name: "state-recovery-fixture", Platform: PlatformOpenAI,
		Type: AccountTypeAPIKey, Concurrency: 1,
		Credentials: map[string]any{"api_key": "fixture-only", "base_url": "https://example.com"},
		Extra:       map[string]any{"use_responses_api": true, "openai_passthrough": passthrough},
	}
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	SetOpenAIClientTransport(c, OpenAIClientTransportHTTP)
	return svc, c, account, upstream, recorder
}

func TestOpenAIGatewayService_StateRecoveryCrossResource(t *testing.T) {
	svc, c, account, upstream, recorder := newStateRecoveryForward(t, false,
		stateRecoveryResponse(http.StatusBadRequest, crossResourceStateError), stateRecoverySuccess())
	body := []byte(stateRecoveryInput)
	original := bytes.Clone(body)

	result, err := svc.Forward(context.Background(), c, account, body)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, upstream.bodies, 2)
	require.Equal(t, original, body, "recovery must not mutate the caller's canonical bytes")
	first, retry := upstream.bodies[0], upstream.bodies[1]
	require.Equal(t, "msg_history", gjson.GetBytes(first, "input.1.id").String())
	require.Equal(t, "fixture-cipher", gjson.GetBytes(first, "input.0.encrypted_content").String())
	for _, path := range []string{"input.0.id", "input.1.id", "input.2.id", "input.0.encrypted_content"} {
		require.False(t, gjson.GetBytes(retry, path).Exists(), path)
	}
	require.Equal(t, "keep reasoning summary", gjson.GetBytes(retry, "input.0.summary.0.text").String())
	require.Equal(t, "9007199254740993", gjson.GetBytes(retry, "input.1.content.0.nonce").Raw)
	require.Equal(t, "call_shared", gjson.GetBytes(retry, "input.2.call_id").String())
	require.Equal(t, `{"n":9007199254740993}`, gjson.GetBytes(retry, "input.2.arguments").String())
	require.Equal(t, "keep tool result", gjson.GetBytes(retry, "input.3.output").String())
	require.Equal(t, "lookup", gjson.GetBytes(retry, "tools.0.name").String())
	for _, req := range upstream.requests {
		require.Equal(t, "https://example.com/v1/responses", req.URL.String())
		require.Equal(t, "Bearer fixture-only", req.Header.Get("Authorization"))
	}
	require.Equal(t, http.StatusOK, recorder.Code)
	require.Equal(t, "resp_recovered", gjson.Get(recorder.Body.String(), "id").String())
	require.NotContains(t, recorder.Body.String(), "different Azure")
}

func stateRecoveryStreamSuccess() *http.Response {
	resp := stateRecoveryResponse(http.StatusOK, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_recovered\",\"model\":\"gpt-5\",\"status\":\"in_progress\"}}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_recovered\",\"model\":\"gpt-5\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":7,\"output_tokens\":3}}}\n\n")
	resp.Header.Set("Content-Type", "text/event-stream")
	return resp
}

func TestOpenAIGatewayService_StateRecoveryThirdRetry(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for name, errorBody := range map[string]string{"resource": crossResourceStateError, "resourceMasked": maskedResourceStateError, "resourceAlias": strings.ReplaceAll(crossResourceStateError, "Azure", "Microsoft"), "encrypted": encryptedReplayStateError} {
				t.Run(fmt.Sprintf("%s/passthrough=%v/stream=%v", name, passthrough, stream), func(t *testing.T) {
					success := stateRecoverySuccess()
					if stream {
						success = stateRecoveryStreamSuccess()
					}
					responses := []*http.Response{}
					failedBodies := []*passthroughCloseTrackingReadCloser{}
					for range 3 {
						failed := stateRecoveryResponse(http.StatusBadRequest, errorBody)
						tracked := &passthroughCloseTrackingReadCloser{Reader: strings.NewReader(errorBody)}
						failed.Body = tracked
						failedBodies = append(failedBodies, tracked)
						responses = append(responses, failed)
					}
					responses = append(responses, success)
					svc, c, account, upstream, recorder := newStateRecoveryForward(t, passthrough, responses...)
					body, err := sjson.SetBytes([]byte(stateRecoveryInput), "stream", stream)
					require.NoError(t, err)
					result, err := svc.Forward(context.Background(), c, account, body)
					require.NoError(t, err)
					require.NotNil(t, result)
					require.Len(t, upstream.requests, 4, "initial request plus three retries")
					require.Equal(t, 7, result.Usage.InputTokens)
					require.Equal(t, 3, result.Usage.OutputTokens)
					for _, failed := range failedBodies {
						require.True(t, failed.closed)
					}
					for i, req := range upstream.requests {
						require.Equal(t, "https://example.com/v1/responses", req.URL.String())
						require.Equal(t, "Bearer fixture-only", req.Header.Get("Authorization"))
						require.Equal(t, int64(len(upstream.bodies[i])), req.ContentLength)
					}
					for _, retry := range upstream.bodies[1:] {
						require.False(t, gjson.GetBytes(retry, "input.1.id").Exists())
						require.False(t, gjson.GetBytes(retry, "input.0.encrypted_content").Exists())
						require.Equal(t, "call_shared", gjson.GetBytes(retry, "input.2.call_id").String())
						require.Equal(t, "keep tool result", gjson.GetBytes(retry, "input.3.output").String())
					}
					require.Contains(t, recorder.Body.String(), "resp_recovered")
					require.NotContains(t, recorder.Body.String(), "could not be verified")
					require.NotContains(t, recorder.Body.String(), "different Azure")
					require.NotContains(t, recorder.Body.String(), "different ***")
				})
			}
		}
	}
}

func TestOpenAIGatewayService_StateRecoveryExhaustion(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		t.Run(fmt.Sprint(passthrough), func(t *testing.T) {
			responses := []*http.Response{}
			for attempt := 1; attempt <= 4; attempt++ {
				body, err := sjson.Set(crossResourceStateError, "error.message", fmt.Sprintf("The requested item was created under a different Azure OpenAI resource. attempt-%d", attempt))
				require.NoError(t, err)
				responses = append(responses, stateRecoveryResponse(http.StatusBadRequest, body))
			}
			responses = append(responses, stateRecoverySuccess()) // Must never be consumed.
			svc, c, account, upstream, recorder := newStateRecoveryForward(t, passthrough, responses...)
			_, err := svc.Forward(context.Background(), c, account, []byte(stateRecoveryInput))
			require.Error(t, err)
			require.Len(t, upstream.requests, 4)
			require.Equal(t, http.StatusBadRequest, recorder.Code)
			require.Len(t, upstream.responses, 1, "the fifth success response must remain unused")
			if passthrough {
				// Preserve the gateway's existing error-disclosure boundary.
				require.JSONEq(t, `{"error":{"message":"Upstream request failed","type":"upstream_error"}}`, recorder.Body.String())
			} else {
				require.Contains(t, recorder.Body.String(), "attempt-4")
			}
			require.NotContains(t, recorder.Body.String(), "attempt-1")
			require.NotContains(t, recorder.Body.String(), "resp_recovered")
		})
	}
}

func TestOpenAIGatewayService_StateRecoveryPreservesOpaqueHistory(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		for name, patch := range map[string]struct {
			path  string
			value any
		}{
			"previous_response":   {"previous_response_id", "resp_previous"},
			"conversation_id":     {"conversation", "conv_previous"},
			"conversation_object": {"conversation", map[string]any{"id": "conv_previous"}},
			"item_reference":      {"input.5", map[string]any{"type": "item_reference", "id": "msg_stored"}},
			"compaction":          {"input.5", map[string]any{"type": "compaction", "id": "cmp_stored", "encrypted_content": "only-history"}},
		} {
			t.Run(fmt.Sprintf("%s/%v", name, passthrough), func(t *testing.T) {
				svc, c, account, upstream, _ := newStateRecoveryForward(t, passthrough,
					stateRecoveryResponse(http.StatusBadRequest, crossResourceStateError),
					stateRecoveryResponse(http.StatusBadRequest, crossResourceStateError),
					stateRecoveryResponse(http.StatusBadRequest, crossResourceStateError), stateRecoverySuccess())
				body, err := sjson.SetBytes([]byte(stateRecoveryInput), patch.path, patch.value)
				require.NoError(t, err)
				_, err = svc.Forward(context.Background(), c, account, body)
				require.NoError(t, err)
				require.Len(t, upstream.bodies, 4)
				for _, retry := range upstream.bodies[1:] {
					require.Equal(t, upstream.bodies[0], retry, "opaque history must not be silently removed")
				}
			})
		}
	}
}

func TestOpenAIGatewayService_StateRecoveryDoesNotAffectOtherRequests(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		for _, tc := range []struct {
			name   string
			status int
			body   string
		}{
			{"ordinary400", 400, `{"error":{"message":"Invalid parameter"}}`},
			{"itemNotFound", 400, `{"error":{"message":"Item not found"}}`},
			{"resourceNearMiss", 400, `{"error":{"message":"The requested item was created under a different project. Check your OpenAI resource configuration."}}`},
			{"maskedUnauthorized", 401, maskedResourceStateError},
			{"maskedRateLimit", 429, maskedResourceStateError},
			{"nearMiss", 400, `{"error":{"message":"The encrypted content field must be a string"}}`},
			{"unauthorized", 401, encryptedReplayStateError},
			{"rateLimit", 429, encryptedReplayStateError},
			{"upstreamFailure", 502, encryptedReplayStateError},
		} {
			t.Run(fmt.Sprintf("%s/%v", tc.name, passthrough), func(t *testing.T) {
				svc, c, account, upstream, _ := newStateRecoveryForward(t, passthrough,
					stateRecoveryResponse(tc.status, tc.body), stateRecoverySuccess())
				_, err := svc.Forward(context.Background(), c, account, []byte(stateRecoveryInput))
				require.Error(t, err)
				require.Len(t, upstream.requests, 1)
			})
		}
		t.Run(fmt.Sprintf("normal/%v", passthrough), func(t *testing.T) {
			svc, c, account, upstream, _ := newStateRecoveryForward(t, passthrough, stateRecoverySuccess())
			_, err := svc.Forward(context.Background(), c, account, []byte(stateRecoveryInput))
			require.NoError(t, err)
			require.Len(t, upstream.bodies, 1)
			require.Equal(t, "msg_history", gjson.GetBytes(upstream.bodies[0], "input.1.id").String())
			require.Equal(t, "fixture-cipher", gjson.GetBytes(upstream.bodies[0], "input.0.encrypted_content").String())
		})
	}
}

type stateRecoveryCancelBody struct {
	io.Reader
	cancel context.CancelFunc
}

func (b *stateRecoveryCancelBody) Close() error {
	b.cancel()
	return nil
}

func TestOpenAIGatewayService_StateRecoveryCancellation(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		t.Run(fmt.Sprint(passthrough), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			failed := stateRecoveryResponse(http.StatusBadRequest, crossResourceStateError)
			failed.Body = &stateRecoveryCancelBody{Reader: strings.NewReader(crossResourceStateError), cancel: cancel}
			svc, c, account, upstream, _ := newStateRecoveryForward(t, passthrough, failed, stateRecoverySuccess())
			_, err := svc.Forward(ctx, c, account, []byte(stateRecoveryInput))
			require.ErrorIs(t, err, context.Canceled)
			require.Len(t, upstream.requests, 1)
		})
	}
}

func TestOpenAIGatewayService_StateRecoveryNoProviderLatch(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		t.Run(fmt.Sprint(passthrough), func(t *testing.T) {
			svc, c, account, upstream, _ := newStateRecoveryForward(t, passthrough,
				stateRecoveryResponse(400, encryptedReplayStateError), stateRecoverySuccess(), stateRecoverySuccess())
			_, err := svc.Forward(context.Background(), c, account, []byte(stateRecoveryInput))
			require.NoError(t, err)
			next, _ := gin.CreateTestContext(httptest.NewRecorder())
			next.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
			SetOpenAIClientTransport(next, OpenAIClientTransportHTTP)
			_, err = svc.Forward(context.Background(), next, account, []byte(stateRecoveryInput))
			require.NoError(t, err)
			require.Len(t, upstream.bodies, 3)
			require.False(t, gjson.GetBytes(upstream.bodies[1], "input.0.encrypted_content").Exists())
			require.Equal(t, "fixture-cipher", gjson.GetBytes(upstream.bodies[2], "input.0.encrypted_content").String())
		})
	}
}

func TestOpenAIGatewayService_StateRecoveryErrorShapes(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		for name, errorBody := range map[string]string{
			"structured":      `{"error":{"code":"invalid_encrypted_content","message":"invalid encrypted reasoning"}}`,
			"signature":       `{"error":{"code":"thinking_signature_invalid","message":"The encrypted content could not be verified. Reason: Encrypted content could not be decrypted or parsed."}}`,
			"wrapped":         `{"error":{"code":null,"message":"{\"error\":{\"message\":\"The encrypted content could not be verified.\",\"code\":null}}"}}`,
			"caseInsensitive": `{"error":{"message":"THE ENCRYPTED CONTENT could not be VERIFIED."}}`,
		} {
			t.Run(fmt.Sprintf("%s/%v", name, passthrough), func(t *testing.T) {
				svc, c, account, upstream, _ := newStateRecoveryForward(t, passthrough,
					stateRecoveryResponse(400, errorBody), stateRecoveryResponse(400, errorBody),
					stateRecoveryResponse(400, errorBody), stateRecoverySuccess())
				_, err := svc.Forward(context.Background(), c, account, []byte(stateRecoveryInput))
				require.NoError(t, err)
				require.Len(t, upstream.requests, 4)
			})
		}
	}
}

func TestOpenAIGatewayService_StateRecoveryExcludedPaths(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		for _, mode := range []string{"oauth", "compact"} {
			t.Run(fmt.Sprintf("%s/%v", mode, passthrough), func(t *testing.T) {
				svc, c, account, upstream, _ := newStateRecoveryForward(t, passthrough,
					stateRecoveryResponse(400, crossResourceStateError), stateRecoverySuccess())
				if mode == "oauth" {
					account.Type = AccountTypeOAuth
					account.Credentials = map[string]any{"access_token": "fixture-only", "chatgpt_account_id": "fixture-account"}
				} else {
					c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses/compact", nil)
				}
				_, err := svc.Forward(context.Background(), c, account, []byte(stateRecoveryInput))
				require.Error(t, err)
				require.Len(t, upstream.requests, 1)
			})
		}
	}
}

func TestOpenAIGatewayService_StateRecoveryBudgetSurvivesOtherRetries(t *testing.T) {
	for _, passthrough := range []bool{false, true} {
		t.Run(fmt.Sprint(passthrough), func(t *testing.T) {
			svc, c, account, upstream, _ := newStateRecoveryForward(t, passthrough,
				stateRecoveryResponse(400, crossResourceStateError),
				stateRecoveryResponse(400, `{"error":{"code":"unknown_parameter","param":"truncation","message":"Unknown parameter: truncation"}}`),
				stateRecoveryResponse(400, encryptedReplayStateError),
				stateRecoveryResponse(400, `{"error":{"code":"invalid_encrypted_content","message":"The encrypted content could not be verified."}}`),
				stateRecoveryResponse(400, crossResourceStateError), stateRecoverySuccess())
			body, err := sjson.SetBytes([]byte(stateRecoveryInput), "truncation", "auto")
			require.NoError(t, err)
			_, err = svc.Forward(context.Background(), c, account, body)
			require.Error(t, err)
			require.Len(t, upstream.bodies, 5, "one rejected-field retry must not reset the three state retries")
			for _, retry := range upstream.bodies[2:] {
				require.False(t, gjson.GetBytes(retry, "truncation").Exists())
				require.False(t, gjson.GetBytes(retry, "input.0.encrypted_content").Exists())
			}
		})
	}
}

func TestStripOpenAIResponsesInlineState_CustomToolCallAndEmptyReasoning(t *testing.T) {
	body := []byte(`{"model":"gpt-5","input":[{"type":"reasoning","id":"rs_only","encrypted_content":"cipher"},{"type":"custom_tool_call","id":"ctc_1","call_id":"call_c","name":"shell","input":"echo hi"}]}`)
	rewritten, changed, err := stripOpenAIResponsesInlineState(body)
	require.NoError(t, err)
	require.True(t, changed)
	require.Equal(t, int64(1), gjson.GetBytes(rewritten, "input.#").Int(), "剥离后仅剩 type 的 reasoning 项应移除")
	require.Equal(t, "custom_tool_call", gjson.GetBytes(rewritten, "input.0.type").String())
	require.False(t, gjson.GetBytes(rewritten, "input.0.id").Exists())
	require.Equal(t, "call_c", gjson.GetBytes(rewritten, "input.0.call_id").String())
	require.Equal(t, "echo hi", gjson.GetBytes(rewritten, "input.0.input").String())
}
