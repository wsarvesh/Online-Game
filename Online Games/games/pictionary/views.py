from django.shortcuts import render
import random as r
from .forms import *
from django.http import HttpResponse, HttpResponseRedirect
from .models import *
import random as r

# Create your views here.
def home(request):
    # p = request.session['players']
    # col = request.session['colors']
    n = request.session['nop']
    t = request.session['turn']
    cards = ''
    word = ''
    win = [0,"",""]
    x = 0
    if request.method == "POST":
        form = Btnform(request.POST)
        form3 = Cardform(request.POST)
        if form.is_valid():
            if 'correct' in request.POST:
                # request.session['score'][t] = request.session['score'][t]+request.session['dice']
                request.session['player_data'][t][0] = request.session['player_data'][t][0]+request.session['dice']

                if request.session['player_data'][t][0] >= 25:
                    win[0] = 1
                    win[1] = request.session['player_data'][t][1]
                    win[2] = request.session['player_data'][t][2]

                request.session['player_data'][t][3] = request.session['player_data'][t][0]*4
                # request.session['dice'] = 0
                request.session['turn'] = (request.session['turn'] + 1)%n
                request.session['dice'] = r.choice([i for i in range(1,7)])
            elif 'wrong' in request.POST:
                # request.session['score'][t] = request.session['score'][t]
                request.session['player_data'][t][0] = request.session['player_data'][t][0]
                request.session['player_data'][t][3] = request.session['player_data'][t][0]*4
                # request.session['dice'] = 0
                request.session['turn'] = (request.session['turn'] + 1)%n
                request.session['dice'] = r.choice([i for i in range(1,7)])
            request.session['valid'] = 0;


        if form3.is_valid():
            if 'showcard' in request.POST:
                c = Cards.objects.all()
                c_no1 = r.choice(request.session['card_index'])
                request.session['card_index'].remove(c_no1)
                c_no2 = r.choice(request.session['card_index'])
                request.session['card_index'].remove(c_no2)
                cards = [[c[c_no1].card_title,c[c_no1].card_object,c[c_no1].card_action,c[c_no1].card_food,c[c_no1].card_allplay],[c[c_no2].card_title,c[c_no2].card_object,c[c_no2].card_action,c[c_no2].card_food,c[c_no2].card_allplay]]
                x = (request.session['player_data'][t][0]+request.session['dice']-1)%5
                word = [cards[0][x],cards[1][x]]
                request.session['valid'] = 2;



        # score = request.session['score']
        # pd = request.session['player_data']

    else:
        form = Btnform()
        request.session['score'] = [0 for i in range(request.session['nop'])]
        score = request.session['score']
        request.session['dice'] = r.choice([i for i in range(1,7)])
    dice = request.session['dice']
    t = request.session['turn']
    pd = request.session['player_data']
    p_turn = pd[t][1]
    p_col = pd[t][2]
    # valid = 1
    valid = request.session['valid']
    gen = request.session['genre'][x]
    p_dict = {
        'dice':dice,
        'form':form,
        'turn':t,
        'valid':valid,
        'Word':word,
        'pd' : pd,
        'win' : win,
        'p_turn':p_turn,
        'p_col':p_col,
        'gen':gen}
    return render(request,'pictionary/home.html',p_dict)

def login(request):
    if request.method == "POST":
        login = Loginform(request.POST)
        players = []
        colors = []
        if login.is_valid():
            if login.cleaned_data['team1']:
                players.append(login.cleaned_data['team1'])
                colors.append('red')
            if login.cleaned_data['team2']:
                players.append(login.cleaned_data['team2'])
                colors.append('blue')
            if login.cleaned_data['team3']:
                players.append(login.cleaned_data['team3'])
                colors.append('green')
            if login.cleaned_data['team4']:
                players.append(login.cleaned_data['team4'])
                colors.append('yellow')
            if login.cleaned_data['team5']:
                players.append(login.cleaned_data['team5'])
                colors.append('orange')
            if login.cleaned_data['team6']:
                players.append(login.cleaned_data['team6'])
                colors.append('pink')

            # request.session['players'] = players
            # request.session['colors'] = colors
            nop = len(players)
            request.session['valid'] = 0
            request.session['nop'] = nop
            # request.session['score'] = [0 for i in range(request.session['nop'])]
            request.session['turn'] = 0
            request.session['dice'] = 0
            request.session['player_data'] = []
            request.session['genre'] = ['Animals & Places','Objects','Actions','Food','Random']
            for i in range(nop):
                request.session['player_data'].append([0,players[i],colors[i],0])
            request.session['card_index'] = [i for i in range(Cards.objects.all().count())]
            if nop == 0:
                return HttpResponseRedirect('/pictionary/')
        return HttpResponseRedirect('/pictionary/board/')
    else:
        login = Loginform()
        return render(request,'pictionary/login.html')

def canvas(request):
    return render(request,'pictionary/canvas.html')

def rules(request):
    return render(request,'pictionary/rules.html')
