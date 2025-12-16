/*Drop in function for Primo customization package*/

(function () {
   "use strict";
   'use strict';

   var app = angular.module('viewCustom', ['angularLoad']);

   /* JQUERY */
   /* This code adds jQuery, which is required for some customizations */
   var jQueryScript = document.createElement("script");
   jQueryScript.src = "https://code.jquery.com/jquery-3.3.1.min.js";
   document.getElementsByTagName("head")[0].appendChild(jQueryScript);

/* SEARCH AI */
app.component('prmSearchBarAfter', {
   bindings: { parentCtrl: '<' },
   controller: 'prmSearchBarAfterController',
});

app.controller('prmSearchBarAfterController', [function() {
   var vm = this;

   this.$onInit = function() {
      try {
         // === AI global loading veil setup ============================
         (function setupAIVeil(){
            if (!document.getElementById('ai-global-veil')) {
               const veil = document.createElement('div');
               veil.id = 'ai-global-veil';
               veil.setAttribute('role','status');
               veil.setAttribute('aria-live','polite');
               veil.innerHTML =
                  '<div class="ai-veil-box">' +
                     '<div class="ai-spinner" aria-hidden="true"></div>' +
                     '<div class="ai-veil-text">Preparing your AI results…</div>' +
                  '</div>';
               document.body.appendChild(veil);
            }
            
            // Global API for veil control
            window.aiVeil = {
               show(msg){
                  const v = document.getElementById('ai-global-veil');
                  if (!v) return;
                  if (msg) v.querySelector('.ai-veil-text').textContent = msg;
                  v.style.display = 'flex'; 
               },
               hide(){
                  const v = document.getElementById('ai-global-veil');
                  if (!v) return;
                  v.style.display = 'none';
               }
            };
         })();

         (function aiCloneOverPrimoBar() {
            let cloneEl = null;
            const AI_ENDPOINT = [YOURPHPENDPOINT];
            function submitAI(query) {
               if (!query.trim()) return;
               
               aiVeil?.show('Preparing your AI results…');

               const formData = new FormData();
               formData.append('query', query);

               fetch(AI_ENDPOINT, { method: 'POST', body: formData })
                  .then(r => r.text())
                  .then(url => {
                     url = (url || '').trim();
                     if (!/^https?:\/\//.test(url)) throw new Error('Invalid redirect URL');
                     aiVeil?.show('Opening results…');
                     window.location.href = url;
                  })
                  .catch(err => {
                     console.error('AI search failed, falling back to standard search:', err);
                     aiVeil?.hide();
                     // Fallback to standard search
                     const nativeInput = document.querySelector('prm-search-bar input#searchBar, prm-search-bar input[aria-label="Search field"]');
                     if (nativeInput) {
                        nativeInput.value = query;
                        const form = nativeInput.closest('form');
                        if (form) form.submit();
                     }
                  });
            }

            function ensureToggle(){
               const stale = document.querySelector('#ai-toggle input[type="checkbox"]');
               if (stale) stale.closest('#ai-toggle').remove();
               const advanced =
                  document.querySelector('prm-search-bar button.switch-to-advanced[aria-label="Advanced Search"]') ||
                  document.querySelector('prm-search-bar button.switch-to-advanced') ||
                  document.querySelector('prm-search-bar a[aria-label="Advanced Search"]') ||
                  document.querySelector('prm-search-bar a.switch-to-advanced') ||
                  document.querySelector('prm-search-bar [data-automation-id="advanced-search-link"]');
               const advancedWrap =
                  document.querySelector('prm-search-bar .search-switch-buttons') ||
                  document.querySelector('prm-search-bar .advanced-search');
               let label = document.getElementById('ai-toggle');
               if (!label) {
                  label = document.createElement('div');
                  label.id = 'ai-toggle';
                  label.className = 'toggle-switch';
                  label.innerHTML =
                     '<input id="simpleSearchInput" type="radio" name="search-mode" value="simple" checked>' +  
                     '<input id="aiToggleInput" type="radio" name="search-mode" value="ai">' +
                     '<div class="toggle-switch-background">' +
                        '<div class="toggle-switch-handle" title="Turn AI Enhancements on-off"></div>' +
                     '</div>' +
                     '<div class="toggle-labels">' +
                        '<span class="toggle-off" title="Standard keyword search without AI assistance">Basic</span>' +
                        '<span class="toggle-on" title="Search the catalog using natural language">AI Enhanced</span>' +
                     '</div>';
               }
               const toggleBg = label.querySelector('.toggle-switch-background');
               const toggleLabels = label.querySelector('.toggle-labels');
               
               [toggleBg, toggleLabels].forEach(element => {
                  element.addEventListener('click', function(e) {
                     const aiInput = label.querySelector('#aiToggleInput');
                     const simpleInput = label.querySelector('#simpleSearchInput');
                     
                     if (simpleInput.checked) {
                        aiInput.checked = true;
                     } else {
                        simpleInput.checked = true;
                     }
                     const checkedInput = label.querySelector('input:checked');
                     checkedInput.dispatchEvent(new Event('change', { bubbles: true }));
                  });
               });

               // Position toggle next to Advanced Search
               if (advanced && advanced.insertAdjacentElement) {
                  if (label.previousElementSibling !== advanced) {
                     advanced.insertAdjacentElement('afterend', label);
                  }
               } else if (advancedWrap) {
                  if (label.parentElement !== advancedWrap) advancedWrap.appendChild(label);
               } else {
                  // Fallback positioning
                  const fallback = document.querySelector('prm-search-bar .search-element-inner');
                  if (fallback && !label.isConnected) fallback.appendChild(label);
               }

               // font inheritance from existing UI
               try {
                  const ref = advanced || advancedWrap || document.querySelector('prm-search-bar .search-element-inner');
                  if (ref) {
                     const cs = getComputedStyle(ref);
                     label.style.fontFamily = cs.fontFamily;
                     label.style.fontSize = cs.fontSize;
                     label.style.fontWeight = cs.fontWeight;
                     label.style.letterSpacing = cs.letterSpacing;
                     label.style.textTransform = cs.textTransform;
                     label.style.color = cs.color;
                  }
               } catch(_) {}
            }

            function mountClone() {
               if (cloneEl) return;

               const host = document.querySelector('prm-search-bar .simple-search-wrapper');
               const anchor = document.querySelector('prm-search-bar .search-element-inner');
               if (!host || !anchor) return;

               const el = host.cloneNode(true);
               el.id = 'aiPrimoClone';
               el.classList.add('ai-primo-overlay');

               // Wire the cloned input for AI functionality
               const input = el.querySelector('#searchBar') || el.querySelector('input[aria-label="Search field"]');
               if (input) {
                  input.id = 'aiSearchBar';
                  input.setAttribute('placeholder', 'Search with AI…');
                  input.addEventListener('keydown', (e) => {
                     if (e.key === 'Enter') {
                        e.preventDefault();
                        submitAI(input.value.trim());
                     }
                  });
               }
               const form = el.querySelector('form') || el;
               if (form) {
                  form.addEventListener('submit', (e) => {
                     e.preventDefault();
                     const q = input ? input.value.trim() : '';
                     if (q) submitAI(q);
                  }, true);
                  form.setAttribute('onsubmit', '');
               }
               const searchBtn =
                  el.querySelector('button[aria-label="Submit search"]') ||
                  el.querySelector('button[aria-label="Search"]') ||
                  el.querySelector('button[type="submit"]');

               if (searchBtn) {
                  searchBtn.addEventListener('click', (e) => {
                     e.preventDefault();
                     e.stopPropagation();
                     const q = (input ? input.value : '').trim();
                     if (q) submitAI(q);
                  });
                  searchBtn.type = 'button';
               }

               anchor.appendChild(el);
               cloneEl = el;
               setTimeout(() => input && input.focus(), 30);
            }

            function unmountClone() {
               if (!cloneEl) return;
               cloneEl.remove();
               cloneEl = null;
            }

            function wireToggle(){
               ensureToggle();
               const container = document.getElementById('ai-toggle');
               const ai = document.getElementById('aiToggleInput');
               if (!container || !ai) return;
               if (container.dataset.aiBound === '1') return;
               container.dataset.aiBound = '1';
               container.addEventListener('change', () => {
                  if (ai.checked) {
                     mountClone();
                  } else {
                     unmountClone();
                  }
               });
            }

            // Initialize and maintain toggle positioning
            wireToggle();
            const bar = document.querySelector('prm-search-bar');
            if (bar) {
               const mo = new MutationObserver(() => {
                  ensureToggle();
                  wireToggle();
               });
               mo.observe(bar, { childList: true, subtree: true });
            }
         })();

      } catch (err) {
         console.log('Search AI initialization error:', err);
      }
   };
}]);

//Load latest jquery
  app.component('prmTopBarBefore', {
      bindings: {parentCtrl: '<'},
      controller: function () {
        this.$onInit = function () {
          loadScript("//ajax.googleapis.com/ajax/libs/jquery/3.4.0/jquery.min.js", jquery_loaded);
        };
      },
      template: ''
    });

});
