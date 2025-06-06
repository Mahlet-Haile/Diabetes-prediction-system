from django.db import models
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.signals import user_logged_in, user_logged_out

class UserActivity(models.Model):
    user = models.ForeignKey('DiabetesUser', on_delete=models.CASCADE)
    action = models.CharField(max_length=50)
    ip_address = models.GenericIPAddressField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = 'User Activities'

    def __str__(self):
        return f"{self.user.username} - {self.action} at {self.timestamp}"

class AuditLog(models.Model):
    user = models.ForeignKey('DiabetesUser', on_delete=models.SET_NULL, null=True)
    action = models.CharField(max_length=50)
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey('content_type', 'object_id')
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField()
    changes = models.JSONField(null=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.username} - {self.action} - {self.timestamp}"

def log_user_activity(sender, request, user, **kwargs):
    UserActivity.objects.create(
        user=user,
        action='login' if sender == user_logged_in else 'logout',
        ip_address=request.META.get('REMOTE_ADDR')
    )

# Connect signals
user_logged_in.connect(log_user_activity)
user_logged_out.connect(log_user_activity)
