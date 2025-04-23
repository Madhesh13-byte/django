from django.shortcuts import render, get_object_or_404
from blog.models import Post

def allpost(request):
    posts = Post.objects.all()  # Better to use .all() to fetch queryset
    return render(request, 'Posts.html', {'posts': posts})  # key should match template usage

def detail(request, blog_id):
    post = get_object_or_404(Post, pk=blog_id)  # this was incorrectly named 'detail'
    return render(request, 'details.html', {'post': post})  # 'post' should match your template
