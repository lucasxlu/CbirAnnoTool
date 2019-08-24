import json
import os
import sys

from django.http import HttpResponse
from django.shortcuts import render

sys.path.append('../')
from cv.controllers.cbir_controller import upload_and_search


# Create your views here.


def welcome(request):
    """
    welcome page for computer vision welcome
    :param request:
    :return:
    """
    return render(request, 'welcome.html')


def anno_view(request):
    return render(request, 'anno.html')


def cbir(request):
    return upload_and_search(request)


def generate_imgs_txt(request):
    imgs = request.GET.get("imgs", '')
    sku_code = request.GET.get("skuCode", '000000')
    print(sku_code)
    print(imgs)

    if not os.path.exists('cv/static/skutxts'):
        os.makedirs('cv/static/skutxts')

    with open('cv/static/skutxts/{}.txt'.format(sku_code), mode='wt', encoding='utf-8') as f:
        f.write(imgs)
        f.flush()
        f.close()

    # response = HttpResponse(imgs, content_type='text/csv')
    # response['Content-Disposition'] = 'attachment; filename={0}.txt'.format(sku_code)
    # return response

    json_str = json.dumps({'file': '/static/skutxts/{}.txt'.format(sku_code), 'imgs': imgs}, ensure_ascii=False)

    return HttpResponse(json_str)
