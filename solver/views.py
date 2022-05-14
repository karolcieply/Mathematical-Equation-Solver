from django.shortcuts import render

# import http response
from django.http import HttpResponse


def home(request):
    context = {}
    return render(request, "solver/index.html", context)

