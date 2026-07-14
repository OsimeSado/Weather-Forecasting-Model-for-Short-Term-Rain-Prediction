from django.db import models


class SiteVisitor(models.Model):
    ip_address = models.GenericIPAddressField()
    path = models.CharField(max_length=500)
    city_searched = models.CharField(max_length=200, blank=True, default="")
    user_agent = models.TextField(blank=True, default="")
    visited_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-visited_at"]

    def __str__(self):
        return f"{self.ip_address} - {self.path} ({self.visited_at:%Y-%m-%d %H:%M})"
