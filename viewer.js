(function () {
  const PDFJS_SRC = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
  const PDFJS_WORKER = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

  const viewer = document.createElement("div");
  viewer.className = "viewer";
  viewer.hidden = true;
  viewer.innerHTML =
    '<div class="viewer-backdrop" data-close></div>' +
    '<div class="viewer-dialog" role="dialog" aria-modal="true" aria-labelledby="viewer-title">' +
    '<header class="viewer-bar">' +
    '<h2 id="viewer-title">Document</h2>' +
    '<button type="button" class="viewer-close" data-close>Close</button>' +
    "</header>" +
    '<div class="viewer-pages" id="viewer-pages"></div>' +
    "</div>";
  document.body.appendChild(viewer);

  const pages = viewer.querySelector("#viewer-pages");
  const title = viewer.querySelector("#viewer-title");
  const closeButton = viewer.querySelector(".viewer-close");
  let lastFocus = null;
  let session = 0;
  let currentPdf = null;
  let pageObserver = null;
  let pdfjsPromise = null;

  function loadPdfJs() {
    if (window.pdfjsLib) {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
      return Promise.resolve(window.pdfjsLib);
    }
    if (!pdfjsPromise) {
      pdfjsPromise = new Promise(function (resolve, reject) {
        const script = document.createElement("script");
        script.src = PDFJS_SRC;
        script.async = true;
        script.onload = function () {
          if (!window.pdfjsLib) {
            pdfjsPromise = null;
            reject(new Error("PDF viewer failed to load"));
            return;
          }
          window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
          resolve(window.pdfjsLib);
        };
        script.onerror = function () {
          pdfjsPromise = null;
          reject(new Error("PDF viewer failed to load"));
        };
        document.head.appendChild(script);
      });
    }
    return pdfjsPromise;
  }

  function resetDocument() {
    if (pageObserver) {
      pageObserver.disconnect();
      pageObserver = null;
    }
    if (currentPdf) {
      currentPdf.destroy().catch(function () {});
      currentPdf = null;
    }
    pages.replaceChildren();
  }

  function showStatus(message) {
    pages.replaceChildren();
    const status = document.createElement("p");
    status.className = "viewer-status";
    status.textContent = message;
    pages.appendChild(status);
  }

  function paintPage(pdf, pageNumber, holder, cssWidth, token) {
    return pdf.getPage(pageNumber).then(function (page) {
      if (token !== session) return;
      const unscaled = page.getViewport({ scale: 1 });
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const viewport = page.getViewport({
        scale: (cssWidth / unscaled.width) * pixelRatio
      });
      const canvas = document.createElement("canvas");
      canvas.width = Math.floor(viewport.width);
      canvas.height = Math.floor(viewport.height);
      holder.style.aspectRatio = unscaled.width + " / " + unscaled.height;
      holder.replaceChildren(canvas);
      return page.render({
        canvasContext: canvas.getContext("2d"),
        viewport: viewport
      }).promise;
    });
  }

  function showDocument(pdf, token) {
    pages.replaceChildren();
    const cssWidth = Math.max(pages.clientWidth - 24, 280);
    pageObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting || token !== session) return;
          const holder = entry.target;
          if (holder.dataset.rendered === "1") return;
          holder.dataset.rendered = "1";
          pageObserver.unobserve(holder);
          paintPage(pdf, Number(holder.dataset.page), holder, cssWidth, token).catch(function () {
            if (token !== session) return;
            holder.textContent = "This page could not be shown.";
          });
        });
      },
      { root: pages, rootMargin: "800px 0px" }
    );

    return pdf.getPage(1).then(function (firstPage) {
      if (token !== session) return;
      const base = firstPage.getViewport({ scale: 1 });
      for (let i = 1; i <= pdf.numPages; i += 1) {
        const holder = document.createElement("div");
        holder.className = "viewer-page";
        holder.dataset.page = String(i);
        holder.style.aspectRatio = base.width + " / " + base.height;
        pages.appendChild(holder);
        pageObserver.observe(holder);
      }
    });
  }

  function openViewer(fileUrl, label) {
    const token = ++session;
    lastFocus = document.activeElement;
    title.textContent = label || "Document";
    resetDocument();
    showStatus("Loading…");
    viewer.hidden = false;
    document.body.style.overflow = "hidden";
    closeButton.focus();

    loadPdfJs()
      .then(function (pdfjs) {
        if (token !== session) return null;
        return pdfjs.getDocument({ url: fileUrl }).promise;
      })
      .then(function (pdf) {
        if (!pdf || token !== session) {
          if (pdf) pdf.destroy().catch(function () {});
          return;
        }
        currentPdf = pdf;
        return showDocument(pdf, token);
      })
      .catch(function () {
        if (token !== session) return;
        showStatus("This document could not be opened.");
      });
  }

  function closeViewer() {
    if (viewer.hidden) return;
    session += 1;
    viewer.hidden = true;
    resetDocument();
    document.body.style.overflow = "";
    if (lastFocus && typeof lastFocus.focus === "function") lastFocus.focus();
  }

  document.addEventListener("click", function (event) {
    const link = event.target.closest("a[data-pdf]");
    if (link) {
      event.preventDefault();
      const heading = link.querySelector("h3");
      const fileUrl = new URL(link.getAttribute("data-pdf"), window.location.href).href;
      openViewer(fileUrl, heading ? heading.textContent.trim() : "Document");
      return;
    }
    if (event.target.closest("[data-close]")) closeViewer();
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeViewer();
  });

  pages.addEventListener("contextmenu", function (event) {
    event.preventDefault();
  });
})();
