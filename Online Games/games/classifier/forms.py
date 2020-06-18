from django import forms

class InputForm(forms.Form):
    file = forms.FileField(required=False)
    demo = forms.CharField(required=False)
