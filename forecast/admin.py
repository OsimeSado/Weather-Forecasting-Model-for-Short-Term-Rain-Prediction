from django.contrib import admin
from .models import SiteVisitor


@admin.register(SiteVisitor)
class SiteVisitorAdmin(admin.ModelAdmin):
    list_display = ("ip_address", "path", "city_searched", "visited_at")
    list_filter = ("visited_at",)
    search_fields = ("ip_address", "city_searched")
    readonly_fields = ("ip_address", "path", "city_searched", "user_agent", "visited_at")
