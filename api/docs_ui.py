"""Swagger UI workaround for JSON text in multipart form fields.

Swagger's client parses JSON-looking strings and serializes primitive arrays
as comma-separated text. Restore the original layers text in its documented
request interceptor, after serialization and before fetch/cURL generation.
"""

from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse


_INTERCEPTOR_JS = """requestInterceptor: function preserveMultipartJson(request) {
      const path = new URL(request.url, location.href).pathname
        .match(/(\\/api\\/(?:nearest|predict))\\/?$/)?.[1];
      if (path && request.body instanceof FormData) {
        const raw = ui.getSystem().oas3Selectors
          .requestBodyValue(path, "post")?.getIn(["layers", "value"]);
        if (typeof raw === "string") request.body.set("layers", raw);
        const dpi = request.body.get("dpi");
        if (typeof dpi === "string" && !dpi.trim()) request.body.delete("dpi");
      }
      return request;
    },
    """


def swagger_docs_html(*, openapi_url: str, title: str) -> HTMLResponse:
    response = get_swagger_ui_html(
        openapi_url=openapi_url,
        title=title,
        swagger_ui_parameters={"showMutatedRequest": True},
    )
    html = response.body.decode("utf-8")
    # The FastAPI helper serializes configuration with JSON; a function passed
    # through swagger_ui_parameters would become an inert string. Inject this
    # executable property at a unique, checked bootstrap anchor instead.
    anchor = "presets: ["
    if html.count(anchor) != 1:
        raise RuntimeError("Swagger UI bootstrap changed; multipart JSON fix needs review")
    return HTMLResponse(html.replace(anchor, _INTERCEPTOR_JS + anchor, 1))
