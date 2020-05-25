from django.db import models

class Cards(models.Model):
    card_title = models.CharField(max_length=100)
    card_object = models.CharField(max_length=100)
    card_action = models.CharField(max_length=100)
    card_food = models.CharField(max_length=100)
    card_allplay = models.CharField(max_length=100)
