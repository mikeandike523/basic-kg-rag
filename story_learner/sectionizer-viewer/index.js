// script.js
document.addEventListener('DOMContentLoaded', () => {
  const listEl = document.getElementById('topics-list');
  const titleEl = document.getElementById('topic-title');
  const parasEl = document.getElementById('topic-paragraphs');
  let sections = [];

  fetch('sections.json')
    .then(res => res.json())
    .then(data => {
      sections = data;
      populateList();
    })
    .catch(err => {
      listEl.innerHTML = '<li>Error loading topics.</li>';
      console.error(err);
    });

  function populateList() {
    listEl.innerHTML = '';
    sections.forEach((sec, idx) => {
      const li = document.createElement('li');
      li.textContent = sec.title;
      li.addEventListener('click', () => selectTopic(idx));
      listEl.appendChild(li);
    });
    // Optionally auto-select the first
    if (sections.length) selectTopic(0);
  }

  function selectTopic(index) {
    // Highlight
    Array.from(listEl.children).forEach((li, i) => {
      li.classList.toggle('active', i === index);
    });
    // Render content
    const sec = sections[index];
    titleEl.textContent = sec.title;
    parasEl.innerHTML = '';
    sec.paragraphs.forEach(p => {
      const pEl = document.createElement('p');
      pEl.textContent = p;
      parasEl.appendChild(pEl);
    });
    // Scroll content to top
    document.getElementById('content').scrollTop = 0;
  }
});
