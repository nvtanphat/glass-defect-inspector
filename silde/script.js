/* Glass Defect Inspector — presentation controller. No build step or network dependency. */
(()=>{'use strict';
const slides=Array.from(document.querySelectorAll('.slide'));
const total=slides.length,deck=document.getElementById('deck');let index=0,touchX=null,touchY=null,animationBusy=false;
const prev=document.getElementById('prev'),next=document.getElementById('next');
const indicator=document.getElementById('slide-indicator');const progress=document.getElementById('progress');
const fit=()=>{const s=Math.min(window.innerWidth/1600,window.innerHeight/900);document.documentElement.style.setProperty('--scale',Math.max(.1,s).toFixed(5));};
window.addEventListener('resize',fit,{passive:true});fit();
const pageFromHash=()=>{const match=location.hash.match(/(?:slide=|\/)(\d+)/);return match?Number(match[1])-1:0;};
function go(i,push=true){const n=Math.max(0,Math.min(total-1,i));if(n===index&&slides[n].classList.contains('active'))return;
 const old=index;index=n;
 slides.forEach((s,j)=>{s.classList.toggle('active',j===index);s.classList.toggle('past',j<index);s.setAttribute('aria-hidden',String(j!==index));});
 indicator.innerHTML=`${String(index+1).padStart(2,'0')} <i>/</i> ${String(total).padStart(2,'0')}`;
 progress.style.width=((index+1)/total*100)+'%';prev.disabled=index===0;next.disabled=index===total-1;
 if(push){history.replaceState(null,'',`#slide=${index+1}`)}
 // Prefer GSAP when loaded, retain offline CSS animation fallback.
 if(window.gsap&&old!==index&& !window.matchMedia('(prefers-reduced-motion: reduce)').matches){
  const el=slides[index];window.gsap.fromTo(el.querySelectorAll('.heading h2, .heading p'),{y:12,opacity:0},{y:0,opacity:1,stagger:.08,duration:.55,ease:'power2.out',clearProps:'transform,opacity'});
 }
 const a=slides[index];a.dispatchEvent(new CustomEvent('slide:visible',{detail:{index:index+1}}));
}
prev.addEventListener('click',()=>go(index-1));next.addEventListener('click',()=>go(index+1));
const overviewPanel=document.getElementById('overview-panel'),overviewGrid=document.getElementById('overview-grid'),lightbox=document.getElementById('lightbox');
const titleFor=s=>s.querySelector('.heading h2')?.innerText||s.querySelector('h1')?.innerText?.replace(/\s+/g,' ')||s.querySelector('.finale-head h2')?.innerText?.replace(/\s+/g,' ')||'Báo cáo';
slides.forEach((s,i)=>{const b=document.createElement('button');b.type='button';b.className='overview-item';b.innerHTML=`<b>${String(i+1).padStart(2,'0')}</b><span>${titleFor(s)}</span>`;b.addEventListener('click',()=>{closeOverview();go(i)});overviewGrid.appendChild(b);});
function closeOverview(){overviewPanel.hidden=true}
function toggleOverview(){overviewPanel.hidden=!overviewPanel.hidden}
document.getElementById('overview').addEventListener('click',toggleOverview);document.getElementById('close-overview').addEventListener('click',closeOverview);
const fs=()=>{if(!document.fullscreenElement){document.documentElement.requestFullscreen?.().catch(()=>{})}else document.exitFullscreen?.().catch(()=>{})};
document.getElementById('fullscreen').addEventListener('click',fs);
document.getElementById('print').addEventListener('click',()=>window.print());
function closeLightbox(){lightbox.hidden=true}
document.getElementById('lightbox-close').addEventListener('click',closeLightbox);
lightbox.addEventListener('click',ev=>{if(ev.target===lightbox)closeLightbox()});
function keydown(e){if(e.altKey||e.ctrlKey||e.metaKey)return;if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))return;
 if(!lightbox.hidden){if(e.key==='Escape'){closeLightbox();e.preventDefault()}return}
 if(!overviewPanel.hidden){if(e.key==='Escape'||e.key.toLowerCase()==='o'){closeOverview();e.preventDefault()}return}
 switch(e.key){case'ArrowRight':case'ArrowDown':case'PageDown':case' ':e.preventDefault();go(index+1);break;case'ArrowLeft':case'ArrowUp':case'PageUp':e.preventDefault();go(index-1);break;case'Home':e.preventDefault();go(0);break;case'End':e.preventDefault();go(total-1);break;case'f':case'F':fs();break;case'p':case'P':window.print();break;case'o':case'O':toggleOverview();break;case'Escape':closeOverview();break;}}
window.addEventListener('keydown',keydown);window.addEventListener('hashchange',()=>go(pageFromHash(),false));
deck.addEventListener('touchstart',e=>{if(e.touches.length===1){touchX=e.touches[0].clientX;touchY=e.touches[0].clientY}},{passive:true});
deck.addEventListener('touchend',e=>{if(touchX===null)return;const dx=e.changedTouches[0].clientX-touchX,dy=e.changedTouches[0].clientY-touchY;touchX=null;touchY=null;if(Math.abs(dx)>50&&Math.abs(dx)>Math.abs(dy)*1.2)go(index+(dx<0?1:-1))},{passive:true});
// Keep original report image source, let presenter change the displayed confusion matrix.
const cmAlt={12:'FBDNet-M',13:'Adaptive Heterogeneous Triad',14:'D-FINE-M',15:'YOLO26m ROI-VFL',16:'Distilled Student YOLO26m'};
const cmMain=document.getElementById('cm-main');
document.querySelectorAll('.cm-tab').forEach(b=>b.addEventListener('click',()=>{const id=b.dataset.cm;document.querySelectorAll('.cm-tab').forEach(t=>{t.classList.toggle('active',t===b);t.setAttribute('aria-selected',String(t===b))});cmMain.src=`assets/image${id}.png`;cmMain.alt=`Ma trận nhầm lẫn ${cmAlt[id]}, ảnh trong báo cáo`;}));
document.querySelector('.cm-display').addEventListener('click',()=>{document.getElementById('lightbox-image').src=cmMain.src;lightbox.hidden=false;});
// Accessible scatter plot hover tooltip. All 14 points contain exact values.
const tooltip=document.getElementById('scatter-tooltip');const plot=document.querySelector('.trade-plot');
document.querySelectorAll('.scatter-svg .point').forEach(p=>{const show=()=>{tooltip.innerHTML=`<b>${p.dataset.model}</b><br>mAP ${p.dataset.ap}% · ${p.dataset.ms} ms<br>FPS ${p.dataset.fps}`;tooltip.style.display='block';};p.addEventListener('pointerenter',show);p.addEventListener('pointermove',e=>{const rect=plot.getBoundingClientRect();tooltip.style.left=Math.min(rect.width-245,Math.max(0,e.clientX-rect.left+10))+'px';tooltip.style.top=Math.max(0,e.clientY-rect.top-81)+'px';});p.addEventListener('pointerleave',()=>{tooltip.style.display='none'})});
// Android/iOS standalone toggle toolbar briefly after a touch.
const tb=document.getElementById('toolbar');document.addEventListener('touchstart',()=>{tb.style.opacity='1'},{passive:true});
// Interactive Demo Inspector Screen Tabs
const demoScreens = {
  workspace: {
    src: 'assets/inspector-workspace.png',
    title: 'Workspace kiểm định lỗi',
    desc: 'Tải ảnh kính 5328×3000 px, crop ROI, suy luận model và đánh dấu bounding box khuyết tật.'
  },
  gradcam: {
    src: 'assets/inspector-gradcam.png',
    title: 'Bản đồ chú ý XAI Grad-CAM',
    desc: 'Hiển thị vùng kích hoạt nơ-ron cục bộ trên từng lỗi đã phát hiện để xác thực quyết định.'
  },
  review: {
    src: 'assets/inspector-review.png',
    title: 'Rà soát & Đánh giá kết quả',
    desc: 'Phân luồng trạng thái Đạt / Không đạt / Cần xác nhận, hỗ trợ QA duyệt và xuất báo cáo kiểm định.'
  },
  dashboard: {
    src: 'assets/inspector-dashboard.png',
    title: 'Dashboard thống kê & Lịch sử',
    desc: 'Theo dõi tổng quan tỷ lệ lỗi, số lượng phôi kính kiểm tra và phân bố khuyết tật theo thời gian.'
  }
};
const demoMain = document.getElementById('demo-main');
const demoTitle = document.getElementById('demo-screen-title');
const demoDesc = document.getElementById('demo-screen-desc');
const demoBtn = document.getElementById('demo-zoom-btn');
document.querySelectorAll('.demo-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const key = btn.dataset.demo;
    if (!demoScreens[key]) return;
    document.querySelectorAll('.demo-tab').forEach(b => {
      const active = b === btn;
      b.classList.toggle('active', active);
      b.setAttribute('aria-selected', String(active));
    });
    const info = demoScreens[key];
    if (demoMain) { demoMain.src = info.src; demoMain.alt = info.title; }
    if (demoTitle) demoTitle.textContent = info.title;
    if (demoDesc) demoDesc.textContent = info.desc;
    if (demoBtn) demoBtn.dataset.figureZoom = info.src;
  });
});
go(pageFromHash(),false);window.__glassSlides={go,next:()=>go(index+1),previous:()=>go(index-1),count:total,get current(){return index+1}};
})();
/* Open high resolution unaltered Word figure for detailed architecture inspection. */
(() => {
 const lb=document.getElementById('lightbox');const pic=document.getElementById('lightbox-image');
 document.querySelectorAll('[data-figure-zoom]').forEach(btn=>{
  btn.addEventListener('click',()=>{pic.src=btn.dataset.figureZoom;pic.alt=btn.querySelector('img')?.alt||'Ảnh gốc trong báo cáo';lb.hidden=false;});
 });
})();
