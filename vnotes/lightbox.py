"""Shared lightbox assets for generated note pages."""
from __future__ import annotations


NOTE_LIGHTBOX_MARKER = "data-vnotes-lightbox"

NOTE_LIGHTBOX_ASSETS = r"""
<style data-vnotes-lightbox>
.cover,.media img,.frame-item img,.svg-wrap{cursor:zoom-in}
.vnotes-lightbox{position:fixed;inset:0;z-index:9999;display:none;align-items:center;justify-content:center;background:rgba(12,12,12,.88);padding:24px}
.vnotes-lightbox.open{display:flex}
.vnotes-lightbox figure{margin:0;max-width:96vw;max-height:92vh;display:flex;flex-direction:column;align-items:center;gap:10px}
.vnotes-lightbox img{max-width:96vw;max-height:84vh;object-fit:contain;border-radius:10px;background:#fff;box-shadow:0 24px 80px rgba(0,0,0,.45)}
.vnotes-lightbox figcaption{max-width:min(820px,92vw);color:rgba(255,255,255,.82);font-size:13px;line-height:1.5;text-align:center}
.vnotes-lightbox-close{position:fixed;top:18px;right:18px;width:40px;height:40px;border:1px solid rgba(255,255,255,.28);border-radius:999px;background:rgba(255,255,255,.12);color:#fff;font-size:22px;line-height:1;cursor:pointer;backdrop-filter:blur(10px)}
.vnotes-lightbox-close:hover{background:rgba(255,255,255,.2)}
html.vnotes-lightbox-open{overflow:hidden}
@media (max-width:640px){.vnotes-lightbox{padding:12px}.vnotes-lightbox-close{top:10px;right:10px}.vnotes-lightbox img{max-width:94vw;max-height:80vh}}
</style>
<script data-vnotes-lightbox>
(function(){
  if(window.__vnotesLightboxReady) return;
  window.__vnotesLightboxReady = true;

  function textFrom(node){
    return node ? String(node.textContent || '').replace(/\s+/g, ' ').trim() : '';
  }

  function svgToImageSource(svg){
    var clone = svg.cloneNode(true);
    if(!clone.getAttribute('xmlns')) clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(new XMLSerializer().serializeToString(clone));
  }

  function sourceFromTarget(target){
    if(!target || !target.closest) return null;

    var img = target.closest('img');
    if(img && (img.classList.contains('cover') || img.closest('.hd-cover,.media,.frame-gallery,.frame-item'))){
      var fig = img.closest('figure');
      return {
        src: img.currentSrc || img.src,
        caption: textFrom(fig ? fig.querySelector('figcaption') : null) || img.alt || ''
      };
    }

    var wrap = target.closest('.svg-wrap');
    if(wrap){
      var svg = wrap.querySelector('svg');
      if(svg){
        var figure = wrap.closest('figure');
        return {
          src: svgToImageSource(svg),
          caption: textFrom(figure ? figure.querySelector('figcaption') : null)
        };
      }
    }
    return null;
  }

  var overlay = document.createElement('div');
  overlay.className = 'vnotes-lightbox';
  overlay.setAttribute('aria-hidden', 'true');
  overlay.innerHTML = '<button type="button" class="vnotes-lightbox-close" aria-label="Close">&#215;</button><figure><img alt=""><figcaption></figcaption></figure>';
  var image = overlay.querySelector('img');
  var caption = overlay.querySelector('figcaption');

  function close(){
    overlay.classList.remove('open');
    overlay.setAttribute('aria-hidden', 'true');
    document.documentElement.classList.remove('vnotes-lightbox-open');
    image.removeAttribute('src');
  }

  function open(src, text){
    image.src = src;
    caption.textContent = text || '';
    overlay.classList.add('open');
    overlay.setAttribute('aria-hidden', 'false');
    document.documentElement.classList.add('vnotes-lightbox-open');
  }

  overlay.addEventListener('click', function(event){
    if(event.target === overlay || event.target.closest('.vnotes-lightbox-close')) close();
  });
  document.addEventListener('keydown', function(event){
    if(event.key === 'Escape' && overlay.classList.contains('open')) close();
  });
  document.addEventListener('click', function(event){
    if(event.defaultPrevented) return;
    var source = sourceFromTarget(event.target);
    if(!source || !source.src) return;
    event.preventDefault();
    open(source.src, source.caption);
  });

  document.addEventListener('DOMContentLoaded', function(){
    if(!overlay.parentNode) document.body.appendChild(overlay);
  });
  if(document.body) document.body.appendChild(overlay);
})();
</script>
"""


def inject_note_lightbox(html_doc: str) -> str:
    """Inject the note-page lightbox unless it is already present."""
    if NOTE_LIGHTBOX_MARKER in html_doc:
        return html_doc
    marker = "</body>"
    lower = html_doc.lower()
    idx = lower.rfind(marker)
    if idx < 0:
        return html_doc + NOTE_LIGHTBOX_ASSETS
    return html_doc[:idx] + NOTE_LIGHTBOX_ASSETS + html_doc[idx:]
