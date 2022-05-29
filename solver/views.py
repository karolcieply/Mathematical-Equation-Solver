from django.shortcuts import render, redirect

from solver.models import Prediction
from .forms import PredictionForm


def home(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
  
        if form.is_valid():
            form.save()
            return redirect('result')
    else:
        form = PredictionForm()
    return render(request, 'solver/index.html', {'form' : form})

def result(request):
    # if request.method == "GET":
        # todo: get the prediction from the model
        # digit_drawed = Prediction.objects.last().drawed_digit

    return render(request, 'solver/result.html')