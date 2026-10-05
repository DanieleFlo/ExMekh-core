(function () {
    if (window.__exmekhAgentBoard) return;
    window.__exmekhAgentBoard = true;
    const scopeFor = element => element.closest('.agent-workbench')?.dataset.scope;
    const scrollPositions = new Map();
    document.addEventListener('scroll', event => {
        const viewport = event.target;
        if (!viewport.isConnected || !viewport.matches?.('.agent-board-viewport')) return;
        const scope = scopeFor(viewport);
        if (scope) scrollPositions.set(scope, {left: viewport.scrollLeft, top: viewport.scrollTop});
    }, true);
    const restoreScroll = viewport => {
        const position = scrollPositions.get(scopeFor(viewport));
        if (!position) return;
        viewport.scrollLeft = position.left;
        viewport.scrollTop = position.top;
    };
    const observer = new MutationObserver(records => {
        for (const record of records) {
            for (const node of record.addedNodes) {
                if (node.nodeType !== Node.ELEMENT_NODE) continue;
                if (node.matches('.agent-board-viewport')) restoreScroll(node);
                node.querySelectorAll('.agent-board-viewport').forEach(restoreScroll);
            }
        }
    });
    observer.observe(document.documentElement, {childList: true, subtree: true});
    const send = (element, action, data) => {
        const scope = scopeFor(element);
        if (scope && typeof emitEvent === 'function') emitEvent('agent_board_action', {scope, action, ...data});
    };
    let drag = null;
    let suppressClick = false;
    document.addEventListener('pointerdown', event => {
        if (event.button !== 0 || !event.isPrimary) return;
        const node = event.target.closest('.agent-board-node');
        const viewport = event.target.closest('.agent-board-viewport');
        if (!viewport) return;
        suppressClick = false;
        if (node) {
            drag = {node, target: node, pointer: event.pointerId, x: event.clientX, y: event.clientY,
                left: parseFloat(node.style.left), top: parseFloat(node.style.top), moved: false};
        } else if (event.target.closest('.agent-board')) {
            drag = {viewport, target: viewport, pointer: event.pointerId, x: event.clientX, y: event.clientY,
                left: viewport.scrollLeft, top: viewport.scrollTop, moved: false};
            viewport.classList.add('is-panning');
            viewport.focus({preventScroll: true});
        } else {
            return;
        }
        drag.target.setPointerCapture(event.pointerId);
        event.preventDefault();
    });
    document.addEventListener('pointermove', event => {
        if (!drag || drag.pointer !== event.pointerId) return;
        const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) < 5 && !drag.moved) return;
        drag.moved = true;
        if (drag.node) {
            drag.node.style.left = Math.max(0, drag.left + dx) + 'px';
            drag.node.style.top = Math.max(0, drag.top + dy) + 'px';
        } else {
            drag.viewport.scrollLeft = drag.left - dx;
            drag.viewport.scrollTop = drag.top - dy;
        }
    });
    const finishDrag = event => {
        if (!drag || drag.pointer !== event.pointerId) return;
        const completed = drag;
        drag = null;
        suppressClick = completed.moved;
        completed.viewport?.classList.remove('is-panning');
        if (completed.target.hasPointerCapture(event.pointerId)) {
            completed.target.releasePointerCapture(event.pointerId);
        }
        if (completed.node && completed.moved) {
            if (event.type === 'pointerup') {
                send(completed.node, 'move', {id: completed.node.dataset.node,
                    x: parseFloat(completed.node.style.left), y: parseFloat(completed.node.style.top)});
            } else {
                completed.node.style.left = completed.left + 'px';
                completed.node.style.top = completed.top + 'px';
            }
        }
    };
    document.addEventListener('pointerup', finishDrag);
    document.addEventListener('pointercancel', finishDrag);
    document.addEventListener('lostpointercapture', finishDrag);
    const pan = (viewport, direction) => {
        const amount = Math.max(100, viewport.clientWidth * 0.6);
        viewport.scrollBy({
            left: direction === 'left' ? -amount : direction === 'right' ? amount : 0,
            top: direction === 'up' ? -160 : direction === 'down' ? 160 : 0,
        });
    };
    document.addEventListener('keydown', event => {
        if (!event.target.matches('.agent-board-viewport')) return;
        const directions = {ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down'};
        if (directions[event.key]) {
            event.preventDefault();
            pan(event.target, directions[event.key]);
        }
    });
    document.addEventListener('click', event => {
        const control = event.target.closest('[data-board-pan]');
        if (control) {
            pan(control.closest('.agent-board-area').querySelector('.agent-board-viewport'),
                control.dataset.boardPan);
            return;
        }
        if (suppressClick) {
            suppressClick = false;
            if (event.detail > 0 && event.target.closest('.agent-board-viewport')) return;
        }
        const node = event.target.closest('.agent-board-node');
        if (node) send(node, 'select', {id: node.dataset.node});
        const item = event.target.closest('.agent-palette-item');
        if (item) send(item, 'add', {id: item.dataset.catalog});
    });
    document.addEventListener('dragstart', event => {
        const item = event.target.closest('.agent-palette-item');
        if (item) event.dataTransfer.setData('application/exmekh-capability', item.dataset.catalog);
    });
    document.addEventListener('dragover', event => {
        if (event.target.closest('.agent-board') && Array.from(event.dataTransfer.types).includes('application/exmekh-capability')) event.preventDefault();
    });
    document.addEventListener('drop', event => {
        const board = event.target.closest('.agent-board');
        const id = event.dataTransfer.getData('application/exmekh-capability');
        if (!board || !id) return;
        event.preventDefault();
        const bounds = board.getBoundingClientRect();
        send(board, 'add', {id, x: event.clientX - bounds.left, y: event.clientY - bounds.top});
    });
})();