from django.db import connections
from django.db import models

# Create your models here.

# class MyTest(models.Model):
#     L1_Engine = models.CharField(max_length=200, primary_key=True)
#     Primary_Execution_Team = models.CharField(max_length=150)
#     Description = models.CharField(max_length=500)
#     class Meta:
#         db_table = "mytest"



class MyTest(models.Model):
    id = models.AutoField(primary_key=True )
    L1_Engine = models.CharField(max_length=200)
    Primary_Execution_Team = models.CharField(max_length=150)
    Description = models.CharField(max_length=500)

    class Meta:
        db_table = "mytest"


# Provide a one-off default value for the id field
# DEFAULT_ID = 0  # You can choose any starting value you prefer
# MyTest._meta.get_field('id').default = DEFAULT_ID
