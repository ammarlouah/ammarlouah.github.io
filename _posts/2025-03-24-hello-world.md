---
title: "Welcome to my Blog"
date: 2025-03-24
categories: [Blogging]
tags: [blog]
---

Hi there and welcome to my blog!

Here, I'll share everything related to me, my projects, learnings, and everything I discover in the world of technology, artificial intelligence, and software development.

I hope you enjoy the content. See you soon!

¡Disfruten!

<p>Visitor count: <span id="visitor-count"></span></p>

<script>
  fetch('https://api.countapi.xyz/update/ammar_blog/welcome?amount=1')
    .then(res => res.json())
    .then(data => {
      document.getElementById('visitor-count').innerText = data.value;
    });
</script>

---
