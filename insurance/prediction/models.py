from django.db import models

class InsuranceData(models.Model):
    IDpol = models.CharField(max_length=50, primary_key=True)
    ClaimNb = models.IntegerField()  # This will be our target variable (0 or 1)
    Exposure = models.FloatField()
    Area = models.CharField(max_length=50)
    VehPower = models.IntegerField()
    VehAge = models.IntegerField()
    DrivAge = models.IntegerField()
    BonusMalus = models.IntegerField()
    VehBrand = models.CharField(max_length=50)
    VehGas = models.CharField(max_length=20)
    Density = models.FloatField()
    Region = models.CharField(max_length=50)
    
    def __str__(self):
        return f"Policy {self.IDpol}"