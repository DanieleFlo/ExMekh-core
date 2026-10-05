/**
 * Smart Chat Auto-Scroll Controller for ExMekh Agent.
 *
 * Behavior:
 * - By default (`pinned = true`), automatically scrolls `.chat-scroll-container`
 *   downward as content grows (streaming tokens, reasoning blocks, skills, tools).
 * - If the user scrolls upward away from the bottom, `pinned` becomes `false`
 *   and automatic downward scrolling stops so the user's reading position stays fixed.
 * - If the user subsequently scrolls all the way back down to the bottom, `pinned`
 *   becomes `true` again and automatic downward scrolling resumes.
 */
(function () {
    if (window.__exmekhChatScroll) return;

    const BOTTOM_THRESHOLD = 24;

    function getDistanceFromBottom(el) {
        return Math.max(0, el.scrollHeight - el.clientHeight - el.scrollTop);
    }

    function isContainerVisible(el) {
        return Boolean(
            el &&
            el.clientHeight > 0 &&
            el.offsetParent !== null &&
            window.getComputedStyle(el).display !== 'none' &&
            window.getComputedStyle(el).visibility !== 'hidden'
        );
    }

    function refreshFloatingButton() {
        const btn = document.getElementById('exmekh-scroll-to-bottom-btn');
        if (!btn) return;
        const containers = Array.from(document.querySelectorAll('.chat-scroll-container'));
        const active = containers.find(isContainerVisible);
        if (!active || !active.__exmekhScrollState) {
            btn.style.opacity = '0';
            btn.style.pointerEvents = 'none';
            btn.style.transform = 'translateY(8px)';
            btn.setAttribute('data-visible', 'false');
            return;
        }
        const state = active.__exmekhScrollState;
        const shouldShow = !state.pinned && (active.scrollHeight - active.clientHeight > BOTTOM_THRESHOLD);
        btn.style.opacity = shouldShow ? '1' : '0';
        btn.style.pointerEvents = shouldShow ? 'auto' : 'none';
        btn.style.transform = shouldShow ? 'translateY(0)' : 'translateY(8px)';
        btn.setAttribute('data-visible', shouldShow ? 'true' : 'false');
    }

    function updateActivePoiHighlight(active, items, track) {
        if (!active || !track) return;
        const dots = Array.from(track.querySelectorAll('.chat-poi-dot'));
        if (dots.length === 0 || !items || dots.length !== items.length) return;

        const containerRect = active.getBoundingClientRect();
        const viewportCenter = containerRect.top + containerRect.height / 2;

        let closestIdx = 0;
        let minDistance = Infinity;

        items.forEach((item, idx) => {
            const rect = item.el.getBoundingClientRect();
            const elCenter = rect.top + rect.height / 2;
            const dist = Math.abs(elCenter - viewportCenter);
            if (dist < minDistance) {
                minDistance = dist;
                closestIdx = idx;
            }
        });

        dots.forEach((dot, idx) => {
            dot.classList.toggle('active', idx === closestIdx);
        });
    }

    function refreshPointsOfInterest() {
        const track = document.getElementById('exmekh-chat-poi-track');
        if (!track) return;

        const containers = Array.from(document.querySelectorAll('.chat-scroll-container'));
        const active = containers.find(isContainerVisible);

        if (!active || (active.scrollHeight <= active.clientHeight + 5)) {
            if (track.classList.contains('visible')) {
                track.classList.remove('visible');
                track.innerHTML = '';
                track.dataset.poiKey = '';
                track.__exmekhPoiRows = [];
            }
            return;
        }

        const userNodes = Array.from(active.querySelectorAll('.chat-message-user, .msg-user'));
        const userRows = [...new Set(userNodes.map(el => el.closest('.chat-message-row') || el))];

        const assistantNodes = Array.from(active.querySelectorAll('.chat-message-assistant, .msg-assistant'));
        const assistantRows = [...new Set(assistantNodes.map(el => el.closest('.chat-message-row') || el))];

        if (userRows.length === 0 && assistantRows.length === 0) {
            if (track.classList.contains('visible')) {
                track.classList.remove('visible');
                track.innerHTML = '';
                track.dataset.poiKey = '';
                track.__exmekhPoiRows = [];
            }
            return;
        }

        const items = [];
        userRows.forEach((row, idx) => {
            const textEl = row.querySelector('.msg-user') || row;
            const text = (textEl.textContent || '').trim();
            const snippet = text.length > 36 ? text.slice(0, 36) + '...' : text;
            items.push({
                type: 'user',
                label: snippet ? `👤 Message #${idx + 1}: ${snippet}` : `👤 Message #${idx + 1}`,
                el: row,
            });
        });

        if (assistantRows.length > 0) {
            const finalRow = assistantRows[assistantRows.length - 1];
            items.push({
                type: 'assistant-final',
                label: '✨ Final response',
                el: finalRow,
            });
        }

        const poiKey = items.map(it => it.type + ':' + it.label).join('|');
        const sameRows = track.__exmekhPoiRows && items.every((item, index) => track.__exmekhPoiRows[index] === item.el);
        if (track.dataset.poiKey === poiKey && track.children.length === items.length && sameRows) {
            updateActivePoiHighlight(active, items, track);
            return;
        }

        track.dataset.poiKey = poiKey;
        track.__exmekhPoiRows = items.map(item => item.el);
        track.classList.add('visible');
        track.innerHTML = '';

        items.forEach(item => {
            const dot = document.createElement('div');
            dot.className = `chat-poi-dot ${item.type === 'user' ? 'poi-user' : 'poi-assistant-final'}`;

            const tooltip = document.createElement('div');
            tooltip.className = 'chat-poi-tooltip';
            tooltip.textContent = item.label;
            dot.appendChild(tooltip);

            dot.addEventListener('click', (e) => {
                e.stopPropagation();
                if (active.__exmekhScrollState) {
                    if (item.type === 'assistant-final') {
                        active.__exmekhScrollState.pinned = true;
                        active.__exmekhScrollState.lastUserUpInputAt = 0;
                    } else {
                        active.__exmekhScrollState.pinned = false;
                    }
                }
                item.el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });

            track.appendChild(dot);
        });

        updateActivePoiHighlight(active, items, track);
    }

    function updatePinnedIndicators(el, state) {
        const val = state.pinned ? 'true' : 'false';
        if (el.getAttribute('data-scroll-pinned') !== val) {
            el.setAttribute('data-scroll-pinned', val);
        }
        refreshFloatingButton();
        refreshPointsOfInterest();
    }

    function scrollToBottom(el, state) {
        if (el.clientHeight === 0) return;
        const targetTop = Math.max(0, el.scrollHeight - el.clientHeight);
        state.isProgrammatic = true;
        el.scrollTop = targetTop;
        state.lastScrollTop = el.scrollTop;
        state.lastScrollHeight = el.scrollHeight;
        state.lastClientHeight = el.clientHeight;
        state.savedScrollTop = el.scrollTop;
        updatePinnedIndicators(el, state);
        requestAnimationFrame(() => {
            state.isProgrammatic = false;
        });
    }

    function syncOnContentOrLayoutChange(el, state, hasDirectChildRemoved) {
        const currentResetSeq = el.getAttribute('data-scroll-reset-seq') || '0';
        if (currentResetSeq !== state.lastResetSeq) {
            state.lastResetSeq = currentResetSeq;
            state.pinned = true;
            state.lastUserUpInputAt = 0;
            if (el.clientHeight > 0 && el.children.length > 0) {
                state.wasEmptyOrHidden = false;
                scrollToBottom(el, state);
            }
            updatePinnedIndicators(el, state);
            return;
        }

        if (el.clientHeight === 0 || el.children.length === 0) {
            state.wasEmptyOrHidden = true;
            refreshFloatingButton();
            return;
        }

        if (state.wasEmptyOrHidden || hasDirectChildRemoved) {
            state.wasEmptyOrHidden = false;
            if (state.pinned) {
                scrollToBottom(el, state);
            } else {
                const maxTop = Math.max(0, el.scrollHeight - el.clientHeight);
                state.isProgrammatic = true;
                el.scrollTop = Math.min(state.savedScrollTop, maxTop);
                state.lastScrollTop = el.scrollTop;
                state.lastScrollHeight = el.scrollHeight;
                state.lastClientHeight = el.clientHeight;
                requestAnimationFrame(() => {
                    state.isProgrammatic = false;
                });
                updatePinnedIndicators(el, state);
            }
            return;
        }

        const distFromBottom = getDistanceFromBottom(el);
        const prevMaxTop = Math.max(0, state.lastScrollHeight - state.lastClientHeight);
        const distFromPrevBottom = Math.max(0, prevMaxTop - el.scrollTop);
        const recentlyScrolledUp = (performance.now() - state.lastUserUpInputAt) < 120;

        if (
            el.scrollHeight >= state.lastScrollHeight &&
            el.scrollTop < state.lastScrollTop - 1 &&
            distFromBottom > BOTTOM_THRESHOLD
        ) {
            state.pinned = false;
            state.savedScrollTop = el.scrollTop;
            state.lastScrollTop = el.scrollTop;
            state.lastScrollHeight = el.scrollHeight;
            state.lastClientHeight = el.clientHeight;
            updatePinnedIndicators(el, state);
            return;
        }

        if (
            !state.pinned &&
            !recentlyScrolledUp &&
            el.scrollTop > state.lastScrollTop &&
            (distFromBottom <= BOTTOM_THRESHOLD || distFromPrevBottom <= BOTTOM_THRESHOLD)
        ) {
            state.pinned = true;
        }

        if (state.pinned) {
            scrollToBottom(el, state);
        } else {
            state.lastScrollHeight = el.scrollHeight;
            state.lastClientHeight = el.clientHeight;
            state.lastScrollTop = el.scrollTop;
            updatePinnedIndicators(el, state);
        }
    }

    function initScrollContainer(el) {
        if (el.__exmekhScrollState) return;

        const state = {
            pinned: true,
            lastScrollTop: el.scrollTop || 0,
            lastScrollHeight: el.scrollHeight || 0,
            lastClientHeight: el.clientHeight || 0,
            savedScrollTop: el.scrollTop || 0,
            lastResetSeq: el.getAttribute('data-scroll-reset-seq') || '0',
            isProgrammatic: false,
            lastUserUpInputAt: 0,
            lastTouchY: null,
            wasEmptyOrHidden: (el.clientHeight === 0 || el.children.length === 0),
        };
        el.__exmekhScrollState = state;
        updatePinnedIndicators(el, state);

        el.addEventListener('wheel', (e) => {
            if (el.scrollHeight <= el.clientHeight) return;
            if (e.deltaY < 0) {
                state.lastUserUpInputAt = performance.now();
                state.pinned = false;
                state.savedScrollTop = el.scrollTop;
                updatePinnedIndicators(el, state);
            } else if (e.deltaY > 0) {
                state.lastUserUpInputAt = 0;
                if (getDistanceFromBottom(el) <= BOTTOM_THRESHOLD) {
                    state.pinned = true;
                    updatePinnedIndicators(el, state);
                }
            }
        }, { passive: true });

        el.addEventListener('touchstart', (e) => {
            if (e.touches && e.touches.length > 0) {
                state.lastTouchY = e.touches[0].clientY;
            }
        }, { passive: true });

        el.addEventListener('touchmove', (e) => {
            if (!e.touches || e.touches.length === 0 || state.lastTouchY === null) return;
            const currentY = e.touches[0].clientY;
            const dy = currentY - state.lastTouchY;
            state.lastTouchY = currentY;
            if (dy > 2 && el.scrollHeight > el.clientHeight) {
                state.lastUserUpInputAt = performance.now();
                state.pinned = false;
                state.savedScrollTop = el.scrollTop;
                updatePinnedIndicators(el, state);
            }
        }, { passive: true });

        el.addEventListener('keydown', (e) => {
            if (['ArrowUp', 'PageUp', 'Home'].includes(e.key)) {
                state.lastUserUpInputAt = performance.now();
                state.pinned = false;
                updatePinnedIndicators(el, state);
            } else if (e.key === 'End') {
                state.lastUserUpInputAt = 0;
                state.pinned = true;
                scrollToBottom(el, state);
            }
        }, { passive: true });

        el.addEventListener('scroll', () => {
            if (el.clientHeight === 0 || el.children.length === 0) return;

            const currentTop = el.scrollTop;
            const currentHeight = el.scrollHeight;
            const currentClient = el.clientHeight;
            const distFromBottom = getDistanceFromBottom(el);
            const prevTop = state.lastScrollTop;
            const prevHeight = state.lastScrollHeight;
            const prevMaxTop = Math.max(0, prevHeight - state.lastClientHeight);
            const distFromPrevBottom = Math.max(0, prevMaxTop - currentTop);

            const heightShrank = prevHeight > 0 && currentHeight < prevHeight - 2;
            if (heightShrank) {
                state.lastScrollTop = currentTop;
                state.lastScrollHeight = currentHeight;
                state.lastClientHeight = currentClient;
                if (state.pinned) {
                    scrollToBottom(el, state);
                } else {
                    state.savedScrollTop = currentTop;
                }
                updatePinnedIndicators(el, state);
                return;
            }

            if (currentTop < prevTop - 1 && distFromBottom > BOTTOM_THRESHOLD) {
                state.pinned = false;
                state.savedScrollTop = currentTop;
                state.lastScrollTop = currentTop;
                state.lastScrollHeight = currentHeight;
                state.lastClientHeight = currentClient;
                updatePinnedIndicators(el, state);
                return;
            }

            if (state.isProgrammatic) {
                state.lastScrollTop = currentTop;
                state.lastScrollHeight = currentHeight;
                state.lastClientHeight = currentClient;
                if (state.pinned) {
                    state.savedScrollTop = currentTop;
                }
                updatePinnedIndicators(el, state);
                return;
            }

            const recentlyScrolledUp = (performance.now() - state.lastUserUpInputAt) < 120;
            if (
                !state.pinned &&
                !recentlyScrolledUp &&
                currentTop >= prevTop &&
                (distFromBottom <= BOTTOM_THRESHOLD || (currentTop > prevTop && distFromPrevBottom <= BOTTOM_THRESHOLD))
            ) {
                state.pinned = true;
                scrollToBottom(el, state);
                return;
            } else if (!state.pinned) {
                state.savedScrollTop = currentTop;
            }

            state.lastScrollTop = currentTop;
            state.lastScrollHeight = currentHeight;
            state.lastClientHeight = currentClient;
            updatePinnedIndicators(el, state);
        }, { passive: true });

        const resizeObserver = new ResizeObserver(() => {
            syncOnContentOrLayoutChange(el, state, false);
        });
        resizeObserver.observe(el);
        for (const child of el.children) {
            resizeObserver.observe(child);
        }

        const mutationObserver = new MutationObserver((mutations) => {
            let hasDirectChildRemoved = false;
            for (const m of mutations) {
                if (m.target === el && m.type === 'childList') {
                    if (m.removedNodes && m.removedNodes.length > 0) {
                        hasDirectChildRemoved = true;
                    }
                    if (m.addedNodes) {
                        for (const node of m.addedNodes) {
                            if (node.nodeType === 1) {
                                resizeObserver.observe(node);
                            }
                        }
                    }
                }
            }
            syncOnContentOrLayoutChange(el, state, hasDirectChildRemoved);
        });

        mutationObserver.observe(el, {
            childList: true,
            characterData: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['data-scroll-reset-seq', 'class', 'style'],
        });

        if (el.clientHeight > 0 && el.children.length > 0) {
            scrollToBottom(el, state);
        }
    }

    function scanContainers() {
        const containers = document.querySelectorAll('.chat-scroll-container');
        containers.forEach(initScrollContainer);
        refreshFloatingButton();
        refreshPointsOfInterest();
    }

    function fallbackCopyText(text) {
        try {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.setAttribute('readonly', '');
            textarea.style.position = 'fixed';
            textarea.style.top = '-9999px';
            textarea.style.left = '-9999px';
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
        } catch (_err) {
            // Ignore clipboard fallback errors in restricted environments
        }
    }

    function copyTextToClipboard(text) {
        window.__exmekhLastCopiedCode = text;
        if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
            return navigator.clipboard.writeText(text).catch(() => fallbackCopyText(text));
        }
        fallbackCopyText(text);
        return Promise.resolve();
    }

    function handleCopyCodeBlock(btn, evt) {
        if (!btn) return;
        if (evt) {
            if (evt.__exmekhHandledCopy) return;
            evt.__exmekhHandledCopy = true;
        }
        let text = btn.getAttribute('data-code');
        if (text === null) {
            const wrapper = btn.closest('.code-block-wrapper') || btn.parentElement;
            const codeEl = wrapper ? wrapper.querySelector('code, pre') : null;
            text = codeEl ? (codeEl.textContent || '').replace(/\n$/, '') : '';
        }
        copyTextToClipboard(text);

        const icon = btn.querySelector('.material-icons, .q-icon');
        if (icon) {
            icon.textContent = 'check';
        }
        btn.classList.add('copied');
        btn.setAttribute('data-copied', 'true');
        if (btn.__exmekhCopyResetTimer) {
            clearTimeout(btn.__exmekhCopyResetTimer);
        }
        btn.__exmekhCopyResetTimer = setTimeout(() => {
            if (icon) {
                icon.textContent = 'content_copy';
            }
            btn.classList.remove('copied');
            btn.setAttribute('data-copied', 'false');
        }, 1500);
    }

    window.__exmekhCopyCodeBlock = handleCopyCodeBlock;

    document.addEventListener('click', (e) => {
        if (e.__exmekhHandledCopy) return;
        const target = e.target && typeof e.target.closest === 'function'
            ? e.target.closest('.code-copy-btn')
            : null;
        if (target) {
            handleCopyCodeBlock(target, e);
        }
    });

    window.__exmekhChatScroll = {
        scanContainers,
        refreshPointsOfInterest,
        scrollActiveToBottom() {
            const containers = Array.from(document.querySelectorAll('.chat-scroll-container'));
            for (const el of containers) {
                if (isContainerVisible(el) && el.__exmekhScrollState) {
                    el.__exmekhScrollState.pinned = true;
                    el.__exmekhScrollState.lastUserUpInputAt = 0;
                    scrollToBottom(el, el.__exmekhScrollState);
                }
            }
        },
    };

    window.addEventListener('resize', refreshPointsOfInterest, { passive: true });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', scanContainers);
    } else {
        scanContainers();
    }

    const docObserver = new MutationObserver((mutations) => {
        let shouldScan = false;
        for (const m of mutations) {
            const target = m.target;
            if (target && target.closest && (target.closest('#exmekh-chat-poi-track') || target.closest('#exmekh-scroll-to-bottom-btn'))) {
                continue;
            }
            shouldScan = true;
            break;
        }
        if (shouldScan) {
            scanContainers();
        }
    });
    docObserver.observe(document.documentElement, { childList: true, subtree: true });
})();
