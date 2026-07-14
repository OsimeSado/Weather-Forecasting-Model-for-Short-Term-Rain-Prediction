NEON_APPS = {'forecast'}


class NeonRouter:
    def _use_neon(self, model):
        from django.conf import settings
        if 'neon' not in settings.DATABASES:
            return False
        return model._meta.app_label in NEON_APPS

    def db_for_read(self, model, **hints):
        if self._use_neon(model):
            return 'neon'
        return 'default'

    def db_for_write(self, model, **hints):
        if self._use_neon(model):
            return 'neon'
        return 'default'

    def allow_relation(self, obj1, obj2, **hints):
        return True

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label in NEON_APPS:
            return db == 'neon'
        return db == 'default'
