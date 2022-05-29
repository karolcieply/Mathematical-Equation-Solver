from django.db import models

class Prediction(models.Model):
    """
    Model to store predictions
    """
    id = models.AutoField(primary_key=True)
    drawed_digit = models.ImageField(upload_to='static/images/')
    prediction = models.CharField(max_length=10, null=True)



    def __str__(self):
        return f'{self.id}'

    class Meta:
        verbose_name = 'Prediction'
        verbose_name_plural = 'Predictions'