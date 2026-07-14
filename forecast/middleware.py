from .models import SiteVisitor


class VisitorTrackingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if request.path.startswith("/static/") or request.path.startswith("/admin/"):
            return response

        ip = self._get_client_ip(request)
        city = ""
        if request.method == "POST":
            city = request.POST.get("city", "").strip()

        try:
            SiteVisitor.objects.create(
                ip_address=ip,
                path=request.path,
                city_searched=city,
                user_agent=request.META.get("HTTP_USER_AGENT", ""),
            )
        except Exception:
            pass

        return response

    def _get_client_ip(self, request):
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.META.get("REMOTE_ADDR", "0.0.0.0")
