from django.db import models
class Post(models.Model):
    title = models.CharField(max_length=100)
    public_date=models.DateTimeField()
    image=models.ImageField(upload_to="images/")
    body=models.TextField()
    def summary(self):
        return self.body[:100]
    def pub_date(self):
        return self.public_date.strftime('%b %d,%y')
    def __str__(self):
        return self.title


# Create your models here.
