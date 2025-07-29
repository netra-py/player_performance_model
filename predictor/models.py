from django.db import models

# Create your models here.

class Teams(models.Model):
    team_name = models.CharField(max_length=255,unique=True)

class Players(models.Model):
    team_name = models.CharField(max_length=255,unique=False)
    player_name = models.CharField(max_length=255,unique=False)

class Venue(models.Model):
    venue = models.CharField(max_length=255,unique=True)