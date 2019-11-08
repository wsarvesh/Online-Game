from django import forms


class Btnform(forms.Form):
    check = forms.CharField(widget = forms.HiddenInput(),required=False)

class Rollform(forms.Form):
    check2 = forms.CharField(widget = forms.HiddenInput(),required=False)

class Cardform(forms.Form):
    check3 = forms.CharField(widget = forms.HiddenInput(),required=False)

class Loginform(forms.Form):
    team1 = forms.CharField(required=False,max_length=20,widget=forms.TextInput(attrs={'class': 'form-control'}))
    team2 = forms.CharField(required=False,max_length=20,widget=forms.TextInput(attrs={'class': 'form-control'}))
    team3 = forms.CharField(required=False,max_length=20,widget=forms.TextInput(attrs={'class': 'form-control'}))
    team4 = forms.CharField(required=False,max_length=20,widget=forms.TextInput(attrs={'class': 'form-control'}))
