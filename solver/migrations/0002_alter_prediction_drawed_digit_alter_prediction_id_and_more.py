# Generated by Django 4.0.4 on 2022-05-28 21:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('solver', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prediction',
            name='drawed_digit',
            field=models.ImageField(upload_to='static/images/'),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='prediction',
            field=models.CharField(max_length=10, null=True),
        ),
    ]
