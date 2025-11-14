---
permalink: /
title: "Welcome"
header:
  overlay_color: "#000"
  overlay_filter: 0.35
  overlay_image: "assets/images/hero_banner.png"
  caption: "Debanjan Saha — Data Engineering · Machine Learning · Creativity"
excerpt: "A blend of engineering, research, and creativity."
---

## 👋 Hi, I'm Debanjan

I'm a Senior Data Engineer & Machine Learning practitioner blending distributed systems, AI research, and creative thinking.  
This space captures:

- My latest **AI & ML articles**
- Hands-on **engineering projects**
- A curated **portfolio**
- A living **CV summary**
- Personal interests like **music & creativity**

---

# 📝 Recent Posts

{% assign posts = site.posts | slice: 0, 6 %}

<div class="tiles">
{% for post in posts %}
  <div class="tile">
    <a href="{{ post.url | relative_url }}">
      {% if post.teaser %}
        <img src="{{ post.teaser | relative_url }}" class="tile-img">
      {% endif %}
      <div class="tile-content">
        <h3 class="tile-title">{{ post.title }}</h3>
        <p class="tile-excerpt">{{ post.excerpt | strip_html | truncate: 120 }}</p>
      </div>
    </a>
  </div>
{% endfor %}
</div>

---

# 🧩 Featured Portfolio

Below are selected highlights from my portfolio.

{% capture portfolio_content %}
{% include_relative portfolio.md %}
{% endcapture %}

{{ portfolio_content | markdownify }}

---

# 📄 CV Highlights

Here’s a short summary of who I am and what I do:

<div class="cv-panel">

- **12+ years experience** in Data Engineering & Machine Learning  
- Specializing in **PySpark, Databricks, Azure, AWS, MLOps**  
- Designed and deployed **long-context AI systems** (100K+ tokens)  
- Built **recommendation systems**, **segmentation modeling**, **propensity models**  
- Architected high-volume distributed big data processing systems   
- Strong experience in **teaching, writing, and public speaking** 

[➡ View Full CV](/cv/)
</div>

---

# 🎸 Music & Creative Projects

Outside of engineering, I'm a **guitarist of 15+ years**, music enthusiast, and currently working on:

- **AI-generated NCS music**  
- Audio modeling  
- Combining ML with creativity  

This mix of technical depth and creativity shapes how I approach everything I build.

---

# 🚀 Explore More

<div style="display:flex; gap:12px; flex-wrap:wrap;">
  <a class="btn btn--primary btn--large" href="/blog/">📝 Read Blog</a>
  <a class="btn btn--primary btn--large" href="/projects/">🧠 AI Projects</a>
  <a class="btn btn--primary btn--large" href="/portfolio/">📁 Portfolio</a>
  <a class="btn btn--primary btn--large" href="/cv/">📄 Curriculum Vitae</a>
</div>

---

# 🏷 Browse by Category

{% assign categories = site.categories %}
<div class="categories-grid">
{% for category in categories %}
  <a class="category-tile" href="/categories/#{{ category[0] | slugify }}">
    <span>{{ category[0] }}</span>
  </a>
{% endfor %}
</div>

---
