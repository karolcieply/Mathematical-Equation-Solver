from django.shortcuts import render, redirect

from solver.models import Prediction
from .forms import PredictionForm
from solver.model.solverModel import SolverModel


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
    context = {}
    if request.method == 'GET':
        image_path = str(Prediction.objects.last().drawed_digit)
        sm = SolverModel()
        sm.createModel()
        # sm.fitModel()
        # sm.saveModel()
        sm.loadModel()
        context['digit']=sm.predictUploadedImage(image_path)
        obj = Prediction.objects.last()
        obj.predicted_digit = context['digit']
        obj.save()

    return render(request, 'solver/result.html', context)