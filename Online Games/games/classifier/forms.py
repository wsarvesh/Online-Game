from django import forms

class InputForm(forms.Form):
    file = forms.FileField(required=False)
    demo = forms.CharField(required=False)

class SelectForm(forms.Form):
    end = forms.CharField(required=False)
    attr = forms.CharField(widget=forms.CheckboxSelectMultiple(),required=False)
    classifier = forms.CharField(widget=forms.CheckboxSelectMultiple(),required=False)
    train = forms.CharField(required=False)
    test = forms.CharField(required=False)
    start = forms.CharField(required=False)
    redirect = forms.CharField(required=False)
