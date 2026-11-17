(function () {
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
    '<iframe id="viewer-frame" title="Document"></iframe>' +
    "</div>";
  document.body.appendChild(viewer);

  const frame = viewer.querySelector("#viewer-frame");
  const title = viewer.querySelector("#viewer-title");
  const closeButton = viewer.querySelector(".viewer-close");
  let lastFocus = null;

  function openViewer(href, label) {
    lastFocus = document.activeElement;
    const name = label || "Document";
    title.textContent = name;
    frame.title = name;
    frame.src = href + "#toolbar=0&navpanes=0&scrollbar=0";
    viewer.hidden = false;
    document.body.style.overflow = "hidden";
    closeButton.focus();
  }

  function closeViewer() {
    if (viewer.hidden) return;
    viewer.hidden = true;
    frame.src = "about:blank";
    document.body.style.overflow = "";
    if (lastFocus && typeof lastFocus.focus === "function") lastFocus.focus();
  }

  document.addEventListener("click", function (event) {
    const link = event.target.closest("a[href$='.pdf']");
    if (link) {
      event.preventDefault();
      const heading = link.querySelector("h3");
      openViewer(link.href, heading ? heading.textContent.trim() : "Document");
      return;
    }
    if (event.target.closest("[data-close]")) closeViewer();
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeViewer();
  });
})();
