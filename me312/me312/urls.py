"""
URL configuration for me312 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from django.conf import settings
from django.conf.urls.static import static
import os

def calculate_value(cash_flows, coc):
    value = 0
    for i in range(len(cash_flows)):
        value += cash_flows[i]/(1 + coc)**i
    
    return value
 
@csrf_exempt
@api_view(["POST"])
def optimise(request):
    if request.method == 'POST':
        data = JSONParser().parse(request)
        rf = float(data['rf'])
        beta_unlev = float(data['beta_unlev'])
        T = float(data['tax'])
        ERP = float(data['erp'])
        EBIT = float(data['ebit'])
        e = float(data['equity'])
        initial_coc = float(data['coc'])
        cash_flows = [float(i.strip()) for i in data['cash_flows'].split(',')]
        ds = np.linspace(0,e,1000)

        if e > 40000:

            c = 24.34044026
            k = 2.13334535
            a = -1.41977962
            b = 0.95456256

        else:

            c = 19.70303026
            k = 2.26221553
            a = -2.68064208
            b = 1.12131852 - 0.54

        def min_cod(cod,d):
            return rf + c/(1+np.exp(k*EBIT*100/(cod*d)+a)) + b - cod

        def gradient(d, cod):
            return beta_unlev*ERP*(1-T)/(d+e)-(rf+beta_unlev*(1+(1-T)*d/e)*ERP)*e/(d+e)**2+cod*(1-T)*e/(d+e)**2

        bounds = [(7.5,25)]
        cods=[]
        for d in ds:
            result = differential_evolution(lambda cod : min_cod(cod[0],d)**2, bounds)
            cods.append(result.x[0])

        d = 0
        grad = gradient(d, cods[0])
        for i in range(2000000):
            d -= e*grad
            grad = gradient(d, cods[np.argmin(np.abs(ds-d))])

        coc = (rf + beta_unlev*(1+(1-T)*d/e)*ERP)*(e/(d+e))+cods[np.argmin(np.abs(ds-d))]*(1-T)*(d/(d+e))

        cocs = []
        for i in range(1000):
            cocs.append((rf + beta_unlev*(1+(1-T)*ds[i]/e)*ERP)*(e/(ds[i]+e))+cods[i]*(1-T)*(ds[i]/(ds[i]+e)))
        initial_d = ds[np.argmin(np.abs(np.array(cocs)-initial_coc))]

        plt.title("Variation of Cost of Capital with Debt to Equity Ratio")
        plt.xlabel("Debt to Equity Ratio")
        plt.ylabel("Cost of Capital")
        plt.plot(np.array(ds)/e, cocs)
        path = settings.MEDIA_ROOT+"/optimised.png"
        if os.path.isfile(path):
            os.remove(path) 
        plt.savefig(path)
        plt.clf()

        plt.title("Variation of Value with Debt to Equity Ratio")
        plt.xlabel("Debt to Equity Ratio")
        plt.ylabel("Value")
        values = [calculate_value(cash_flows, i) for i in cocs]
        plt.plot(np.array(ds)/e, values)
        value_path = settings.MEDIA_ROOT+"/value.png"
        if os.path.isfile(value_path):
            os.remove(value_path) 
        plt.savefig(value_path)

        return JsonResponse({"img":settings.MEDIA_URL+"/optimised.png","value_img":settings.MEDIA_URL+"/value.png","d":d,"coc":coc})

urlpatterns = [
    path('admin/', admin.site.urls),
    path('optimise', optimise),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)