document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide icons
    lucide.createIcons();

    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const btnBrowse = document.getElementById("btnBrowse");
    const dropzoneEmpty = document.getElementById("dropzoneEmpty");
    const dropzonePreview = document.getElementById("dropzonePreview");
    const previewImage = document.getElementById("previewImage");
    const btnClear = document.getElementById("btnClear");
    const btnScan = document.getElementById("btnScan");
    const samplesGrid = document.getElementById("samplesGrid");

    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const resultsContent = document.getElementById("resultsContent");

    // Verdict elements
    const verdictBanner = document.getElementById("verdictBanner");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictTitle = document.getElementById("verdictTitle");
    const verdictCaption = document.getElementById("verdictCaption");
    const gaugeFill = document.getElementById("gaugeFill");
    const gaugeNumber = document.getElementById("gaugeNumber");

    // Branch elements
    const semVal = document.getElementById("semVal");
    const semBar = document.getElementById("semBar");
    const artVal = document.getElementById("artVal");
    const artBar = document.getElementById("artBar");

    // Viewer elements
    const sliderView = document.getElementById("sliderView");
    const sideView = document.getElementById("sideView");
    const heatmapView = document.getElementById("heatmapView");
    const imgBefore = document.getElementById("imgBefore");
    const imgAfter = document.getElementById("imgAfter");
    const imageAfterContainer = document.getElementById("imageAfter");
    const sliderDivider = document.getElementById("sliderDivider");
    const sideImgOrig = document.getElementById("sideImgOrig");
    const sideImgOverlay = document.getElementById("sideImgOverlay");
    const pureHeatmapImg = document.getElementById("pureHeatmapImg");
    const opacityRange = document.getElementById("opacityRange");
    const opacityVal = document.getElementById("opacityVal");

    // Meta elements
    const metaTime = document.getElementById("metaTime");
    const metaSize = document.getElementById("metaSize");
    const metaDevice = document.getElementById("metaDevice");
    const btnDownload = document.getElementById("btnDownload");

    // State
    let currentFile = null;
    let currentSamplePath = null;
    let isDraggingSlider = false;

    // Load samples
    loadSamples();

    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                samplesGrid.innerHTML = data.samples.map(sample => `
                    <div class="sample-card" data-file="${sample.file}">
                        <img src="/static/${sample.file}" class="sample-thumb" alt="${sample.name}">
                        <div class="sample-info">
                            <span class="sample-name">${sample.name}</span>
                            <span class="sample-type ${sample.type.includes('AI') ? 'ai' : 'real'}">
                                ${sample.type}
                            </span>
                            <span class="sample-tag">${sample.tag}</span>
                        </div>
                    </div>
                `).join("");

                // Add click handlers
                document.querySelectorAll(".sample-card").forEach(card => {
                    card.addEventListener("click", () => {
                        const filePath = card.getAttribute("data-file");
                        selectSample(filePath, card.querySelector("img").src);
                    });
                });
            } else {
                samplesGrid.innerHTML = `<p style="color: var(--text-dim); font-size: 13px;">No sample images found.</p>`;
            }
        } catch (err) {
            console.error("Failed to load samples", err);
        }
    }

    function selectSample(filePath, imgSrc) {
        currentFile = null;
        currentSamplePath = filePath;
        previewImage.src = imgSrc;
        dropzoneEmpty.classList.add("hidden");
        dropzonePreview.classList.remove("hidden");
        btnScan.disabled = false;
        
        // Pulse scan button
        btnScan.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Drag & Drop
    btnBrowse.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener("click", () => {
        if (!currentFile && !currentSamplePath) {
            fileInput.click();
        }
    });

    ["dragenter", "dragover"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Paste from clipboard
    window.addEventListener("paste", (e) => {
        const items = e.clipboardData?.items;
        if (items) {
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf("image") !== -1) {
                    const blob = items[i].getAsFile();
                    handleFileSelect(blob);
                    break;
                }
            }
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file (JPG, PNG, WEBP).");
            return;
        }
        currentFile = file;
        currentSamplePath = null;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropzoneEmpty.classList.add("hidden");
            dropzonePreview.classList.remove("hidden");
            btnScan.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    btnClear.addEventListener("click", (e) => {
        e.stopPropagation();
        clearInput();
    });

    function clearInput() {
        currentFile = null;
        currentSamplePath = null;
        fileInput.value = "";
        previewImage.src = "";
        dropzonePreview.classList.add("hidden");
        dropzoneEmpty.classList.remove("hidden");
        btnScan.disabled = true;
    }

    // Inspect Action
    btnScan.addEventListener("click", async () => {
        if (!currentFile && !currentSamplePath) return;

        // UI Loading state
        emptyState.classList.add("hidden");
        resultsContent.classList.add("hidden");
        loadingState.classList.remove("hidden");
        btnScan.disabled = true;
        btnScan.innerHTML = `<i data-lucide="loader-2" class="spin"></i><span>Analyzing with CO-SPY...</span>`;
        lucide.createIcons();

        const formData = new FormData();
        if (currentFile) {
            formData.append("image", currentFile);
        }

        try {
            let response;
            if (currentFile) {
                response = await fetch("/api/scan", {
                    method: "POST",
                    body: formData,
                });
            } else {
                response = await fetch("/api/scan", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ sample_path: currentSamplePath }),
                });
            }

            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || "Failed to inspect image");
            }

            renderResults(data);
        } catch (err) {
            alert("Analysis Error: " + err.message);
            emptyState.classList.remove("hidden");
        } finally {
            loadingState.classList.add("hidden");
            btnScan.disabled = false;
            btnScan.innerHTML = `<i data-lucide="scan"></i><span>Inspect with CO-SPY</span>`;
            lucide.createIcons();
        }
    });

    function renderResults(data) {
        resultsContent.classList.remove("hidden");

        // 1. Verdict Banner
        verdictBanner.className = `verdict-banner ${data.is_ai ? "ai" : "real"}`;
        verdictTitle.textContent = data.verdict.toUpperCase();
        verdictCaption.textContent = data.is_ai 
            ? `High synthetic artifacts detected (${data.confidence}% confidence • Winner: ${data.winning_expert})`
            : `Natural photograph characteristics verified (${data.confidence}% confidence • Both ProGAN & SD experts agree)`;

        verdictIcon.setAttribute("data-lucide", data.is_ai ? "alert-triangle" : "shield-check");

        // Dual-Expert Ensemble Badges
        const scoreProganEl = document.getElementById("scoreProgan");
        const scoreSDEl = document.getElementById("scoreSD");
        if (scoreProganEl && data.score_progan !== undefined) {
            scoreProganEl.textContent = `${data.score_progan}%`;
        }
        if (scoreSDEl && data.score_sd !== undefined) {
            scoreSDEl.textContent = `${data.score_sd}%`;
        }

        // 2. Circular Gauge
        // Radius 42 -> Circumference = 2 * PI * 42 ≈ 263.89
        const circumference = 2 * Math.PI * 42;
        const progressOffset = circumference - (data.confidence / 100) * circumference;
        gaugeFill.style.strokeDashoffset = progressOffset;
        animateNumber(gaugeNumber, 0, data.confidence, 1000);

        // 3. Branch Metrics
        semVal.textContent = `${data.sem_score}%`;
        semBar.style.width = `${data.sem_score}%`;
        artVal.textContent = `${data.art_score}%`;
        artBar.style.width = `${data.art_score}%`;

        // 4. Populate Images
        imgBefore.src = data.original_image;
        imgAfter.src = data.overlay_image;
        sideImgOrig.src = data.original_image;
        sideImgOverlay.src = data.overlay_image;
        pureHeatmapImg.src = data.heatmap_image;
        btnDownload.href = data.overlay_image;

        // Reset Slider to 50%
        setSliderPosition(50);

        // 5. Meta Info
        metaTime.textContent = `${data.inference_time_ms} ms`;
        metaSize.textContent = data.image_size;
        metaDevice.textContent = data.device.toUpperCase();

        lucide.createIcons();
    }

    function animateNumber(element, start, end, duration) {
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const val = (start + (end - start) * progress).toFixed(1);
            element.textContent = val;
            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        requestAnimationFrame(update);
    }

    // Interactive Slider Logic
    function setSliderPosition(percent) {
        percent = Math.max(0, Math.min(100, percent));
        imageAfterContainer.style.width = `${percent}%`;
        sliderDivider.style.left = `${percent}%`;
    }

    function onPointerMove(e) {
        if (!isDraggingSlider) return;
        const rect = sliderView.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        const offsetX = clientX - rect.left;
        const percent = (offsetX / rect.width) * 100;
        setSliderPosition(percent);
    }

    sliderDivider.addEventListener("mousedown", () => { isDraggingSlider = true; });
    sliderView.addEventListener("mousedown", (e) => {
        isDraggingSlider = true;
        onPointerMove(e);
    });

    window.addEventListener("mouseup", () => { isDraggingSlider = false; });
    window.addEventListener("mousemove", onPointerMove);

    // Touch support
    sliderDivider.addEventListener("touchstart", () => { isDraggingSlider = true; });
    sliderView.addEventListener("touchstart", (e) => {
        isDraggingSlider = true;
        onPointerMove(e);
    });
    window.addEventListener("touchend", () => { isDraggingSlider = false; });
    window.addEventListener("touchmove", onPointerMove);

    // View Toggles
    const btnViewSlider = document.getElementById("btnViewSlider");
    const btnViewSide = document.getElementById("btnViewSide");
    const btnViewHeatmap = document.getElementById("btnViewHeatmap");
    const toggleBtns = [btnViewSlider, btnViewSide, btnViewHeatmap];

    toggleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            toggleBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const mode = btn.getAttribute("data-view");
            sliderView.classList.add("hidden");
            sideView.classList.add("hidden");
            heatmapView.classList.add("hidden");

            if (mode === "slider") sliderView.classList.remove("hidden");
            else if (mode === "sideBySide") sideView.classList.remove("hidden");
            else if (mode === "heatmap") heatmapView.classList.remove("hidden");
        });
    });

    // Opacity Range Slider
    opacityRange.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        opacityVal.textContent = `${Math.round(val * 100)}%`;
        imgAfter.style.opacity = val;
        sideImgOverlay.style.opacity = val;
    });
});
