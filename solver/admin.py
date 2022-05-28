from django.contrib import admin
from .models import Prediction

admin.site.site_header = "Solver Admin"

class PredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'drawed_digit', 'prediction')
    
admin.site.register(Prediction, PredictionAdmin)