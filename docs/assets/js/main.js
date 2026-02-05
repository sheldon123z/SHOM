/**
 * SHOM Paper Website - Interactive Features
 * Language toggle, scroll animations, copy citation
 */

document.addEventListener('DOMContentLoaded', () => {
  // ============================================
  // LANGUAGE TOGGLE
  // ============================================
  const langButtons = document.querySelectorAll('.lang-btn');
  const body = document.body;

  // Check saved language preference
  const savedLang = localStorage.getItem('preferredLang') || 'en';
  if (savedLang === 'zh') {
    body.classList.add('zh');
    updateLangButtons('zh');
  }

  langButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const lang = btn.dataset.lang;

      if (lang === 'zh') {
        body.classList.add('zh');
      } else {
        body.classList.remove('zh');
      }

      updateLangButtons(lang);
      localStorage.setItem('preferredLang', lang);
    });
  });

  function updateLangButtons(activeLang) {
    langButtons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.lang === activeLang);
    });
  }

  // ============================================
  // SCROLL REVEAL ANIMATIONS
  // ============================================
  const revealElements = document.querySelectorAll('.reveal');

  const revealOnScroll = () => {
    const windowHeight = window.innerHeight;

    revealElements.forEach(element => {
      const elementTop = element.getBoundingClientRect().top;
      const revealPoint = 150;

      if (elementTop < windowHeight - revealPoint) {
        element.classList.add('active');
      }
    });
  };

  window.addEventListener('scroll', revealOnScroll);
  revealOnScroll(); // Initial check

  // ============================================
  // COPY CITATION
  // ============================================
  const copyBtn = document.querySelector('.copy-btn');

  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const citationText = document.querySelector('.citation-text').textContent;

      navigator.clipboard.writeText(citationText).then(() => {
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!';
        copyBtn.style.background = '#2dd4bf';

        setTimeout(() => {
          copyBtn.innerHTML = originalText;
          copyBtn.style.background = '';
        }, 2000);
      }).catch(err => {
        console.error('Failed to copy:', err);
      });
    });
  }

  // ============================================
  // SMOOTH SCROLL FOR ANCHOR LINKS
  // ============================================
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));

      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // ============================================
  // PARALLAX EFFECT ON HERO
  // ============================================
  const hero = document.querySelector('.hero');

  if (hero) {
    window.addEventListener('scroll', () => {
      const scrolled = window.pageYOffset;
      const rate = scrolled * 0.3;
      hero.style.backgroundPositionY = `${rate}px`;
    });
  }

  // ============================================
  // IMAGE LAZY LOADING
  // ============================================
  const images = document.querySelectorAll('img[data-src]');

  const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        img.src = img.dataset.src;
        img.removeAttribute('data-src');
        observer.unobserve(img);
      }
    });
  });

  images.forEach(img => imageObserver.observe(img));

  // ============================================
  // NAVBAR SCROLL EFFECT
  // ============================================
  const langToggle = document.querySelector('.lang-toggle');

  if (langToggle) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 100) {
        langToggle.style.background = 'rgba(10, 22, 40, 0.95)';
      } else {
        langToggle.style.background = 'rgba(10, 22, 40, 0.9)';
      }
    });
  }

  console.log('🔋 SHOM Paper Website Loaded Successfully');
});
