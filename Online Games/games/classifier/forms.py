from django import forms

class InputForm(forms.Form):
    file = forms.FileField(required=False)
    demo = forms.CharField(required=False)

class SelectForm(forms.Form):
    end = forms.CharField()
    attr = forms.CharField(widget=forms.CheckboxSelectMultiple())
    classifier = forms.CharField(widget=forms.CheckboxSelectMultiple())
    train = forms.CharField()
    test = forms.CharField()
