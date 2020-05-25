from django import forms


class Btnform(forms.Form):
    check = forms.CharField(widget = forms.HiddenInput(),required=False)

class Rollform(forms.Form):
    check2 = forms.CharField(widget = forms.HiddenInput(),required=False)

class Cardform(forms.Form):
    check3 = forms.CharField(widget = forms.HiddenInput(),required=False)

class Loginform(forms.Form):
    team1 = forms.CharField(required=False,max_length=20)
    team2 = forms.CharField(required=False,max_length=20)
    team3 = forms.CharField(required=False,max_length=20)
    team4 = forms.CharField(required=False,max_length=20)
    team5 = forms.CharField(required=False,max_length=20)
    team6 = forms.CharField(required=False,max_length=20)
