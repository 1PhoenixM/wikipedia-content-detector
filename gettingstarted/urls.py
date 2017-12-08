from django.conf.urls import include, url

from django.contrib import admin
admin.autodiscover()

import project.views

# Examples:
# url(r'^$', 'gettingstarted.views.home', name='home'),
# url(r'^blog/', include('blog.urls')),

urlpatterns = [
    url(r'^$', project.views.index, name='index'),
    url(r'^classify', project.views.classify, name='classify'),
    url(r'^admin/', admin.site.urls),
]
