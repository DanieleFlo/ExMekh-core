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
    const updateConnections = movedNodes => {
        const board = movedNodes[0]?.closest('.agent-board');
        if (!board) return;
        const movedIds = new Set(movedNodes.map(node => node.dataset.node));
        const nodes = new Map(Array.from(board.querySelectorAll('.agent-board-node'),
            item => [item.dataset.node, item]));
        const width = Number(board.dataset.nodeWidth);
        const halfHeight = Number(board.dataset.nodeHeight) / 2;
        const offset = Number(board.dataset.curveOffset);
        for (const path of board.querySelectorAll('.agent-board-links path')) {
            if (!movedIds.has(path.dataset.parent) && !movedIds.has(path.dataset.child)) continue;
            const parent = nodes.get(path.dataset.parent), child = nodes.get(path.dataset.child);
            if (!parent || !child) continue;
            const startX = parseFloat(parent.style.left) + width;
            const startY = parseFloat(parent.style.top) + halfHeight;
            const endX = parseFloat(child.style.left);
            const endY = parseFloat(child.style.top) + halfHeight;
            path.setAttribute('d', ['M', startX, startY, 'C',
                startX + offset, startY, endX - offset, endY, endX, endY].join(' '));
        }
    };
    const moveNodes = (positions, dx, dy) => {
        const boundedX = Math.max(dx, -Math.min(...positions.map(item => item.left)));
        const boundedY = Math.max(dy, -Math.min(...positions.map(item => item.top)));
        for (const item of positions) {
            item.node.style.left = item.left + boundedX + 'px';
            item.node.style.top = item.top + boundedY + 'px';
        }
        updateConnections(positions.map(item => item.node));
    };
    let drag = null;
    let suppressClick = false;
    const selectedNodes = board => Array.from(board.querySelectorAll('.agent-board-node.selected'));
    const setSelection = (board, ids) => {
        for (const node of board.querySelectorAll('.agent-board-node')) {
            const selected = ids.has(node.dataset.node);
            node.classList.toggle('selected', selected);
            node.setAttribute('aria-pressed', String(selected));
        }
    };
    const selectionIds = board => selectedNodes(board).map(node => node.dataset.node);
    document.addEventListener('contextmenu', event => {
        if (event.target.closest('.agent-board-viewport')) event.preventDefault();
    });
    document.addEventListener('pointerdown', event => {
        if (![0, 2].includes(event.button) || !event.isPrimary) return;
        const node = event.target.closest('.agent-board-node');
        const viewport = event.target.closest('.agent-board-viewport');
        const board = event.target.closest('.agent-board');
        if (!viewport || !board) return;
        suppressClick = false;
        const previous = new Set(selectionIds(board));
        if (event.button === 2) {
            const bounds = board.getBoundingClientRect();
            const rectangle = document.createElement('div');
            rectangle.className = 'agent-board-selection';
            board.append(rectangle);
            drag = {board, viewport, target: viewport, rectangle, previous,
                additive: event.ctrlKey || event.metaKey,
                originX: event.clientX - bounds.left, originY: event.clientY - bounds.top};
        } else if (node) {
            const ids = new Set(previous);
            if (!ids.has(node.dataset.node)) {
                if (!event.ctrlKey && !event.metaKey) ids.clear();
                ids.add(node.dataset.node);
            }
            setSelection(board, ids);
            drag = {board, node, target: node, previous,
                positions: selectedNodes(board).map(item => ({node: item,
                    left: parseFloat(item.style.left), top: parseFloat(item.style.top)}))};
        } else if (event.target.closest('.agent-board')) {
            drag = {board, viewport, target: viewport, previous,
                additive: event.ctrlKey || event.metaKey,
                left: viewport.scrollLeft, top: viewport.scrollTop};
            viewport.classList.add('is-panning');
            viewport.focus({preventScroll: true});
        } else {
            return;
        }
        Object.assign(drag, {pointer: event.pointerId, x: event.clientX, y: event.clientY, moved: false});
        drag.target.setPointerCapture(event.pointerId);
        event.preventDefault();
    });
    document.addEventListener('pointermove', event => {
        if (!drag || drag.pointer !== event.pointerId) return;
        const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) < 5 && !drag.moved) return;
        drag.moved = true;
        if (drag.rectangle) {
            const bounds = drag.board.getBoundingClientRect();
            const x = Math.max(0, Math.min(drag.board.offsetWidth, event.clientX - bounds.left));
            const y = Math.max(0, Math.min(drag.board.offsetHeight, event.clientY - bounds.top));
            const left = Math.min(x, drag.originX), top = Math.min(y, drag.originY);
            const right = Math.max(x, drag.originX), bottom = Math.max(y, drag.originY);
            Object.assign(drag.rectangle.style, {left: left + 'px', top: top + 'px',
                width: (right - left) + 'px', height: (bottom - top) + 'px'});
            const ids = new Set(drag.additive ? drag.previous : []);
            for (const item of drag.board.querySelectorAll('.agent-board-node')) {
                const nx = parseFloat(item.style.left), ny = parseFloat(item.style.top);
                if (nx <= right && nx + item.offsetWidth >= left && ny <= bottom && ny + item.offsetHeight >= top) {
                    ids.add(item.dataset.node);
                }
            }
            setSelection(drag.board, ids);
        } else if (drag.node) {
            moveNodes(drag.positions, dx, dy);
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
        if (completed.rectangle) {
            completed.rectangle.remove();
            if (event.type === 'pointerup') {
                if (!completed.moved) setSelection(completed.board, completed.additive ? completed.previous : new Set());
                send(completed.board, 'select', {ids: selectionIds(completed.board)});
            } else {
                setSelection(completed.board, completed.previous);
            }
        } else if (!completed.node && !completed.moved && !completed.additive && event.type === 'pointerup') {
            setSelection(completed.board, new Set());
            send(completed.board, 'select', {ids: []});
        }
        if (completed.node && completed.moved) {
            if (event.type === 'pointerup') {
                send(completed.node, 'move_many', {ids: selectionIds(completed.board),
                    positions: completed.positions.map(item => ({id: item.node.dataset.node,
                        x: parseFloat(item.node.style.left), y: parseFloat(item.node.style.top)}))});
            } else {
                moveNodes(completed.positions, 0, 0);
                setSelection(completed.board, completed.previous);
            }
        }
        // Click toggling uses the pre-pointerdown selection.
        if (completed.node && !completed.moved) setSelection(completed.board, completed.previous);
    };
    document.addEventListener('pointerup', finishDrag);
    document.addEventListener('pointercancel', finishDrag);
    document.addEventListener('lostpointercapture', finishDrag);
    const pan = (viewport, direction) => {
        const amount = Number(viewport.querySelector('.agent-board').dataset.panStep);
        viewport.scrollBy({
            left: direction === 'left' ? -amount : direction === 'right' ? amount : 0,
            top: direction === 'up' ? -amount : direction === 'down' ? amount : 0,
            behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
        });
    };
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && drag) {
            event.preventDefault();
            finishDrag({pointerId: drag.pointer, type: 'pointercancel'});
            return;
        }
        if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === 'z') {
            const textEditor = event.target.closest('input, textarea, [contenteditable="true"]');
            if (textEditor && !textEditor.closest('.q-select')) return;
            const boards = Array.from(document.querySelectorAll('.agent-workbench'))
                .filter(board => board.getBoundingClientRect().width > 0);
            const board = boards.at(-1);
            if (!board) return;
            const activeDialog = Array.from(document.querySelectorAll('.q-dialog[aria-modal="true"]'))
                .filter(dialog => dialog.getBoundingClientRect().width > 0).at(-1);
            if (activeDialog && !activeDialog.contains(board)) return;
            event.preventDefault();
            if (drag) {
                finishDrag({pointerId: drag.pointer, type: 'pointercancel'});
            } else {
                send(board, 'undo', {});
            }
            return;
        }
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
        if (node) {
            const board = node.closest('.agent-board');
            const ids = new Set(event.ctrlKey || event.metaKey ? selectionIds(board) : []);
            if (ids.has(node.dataset.node)) ids.delete(node.dataset.node);
            else ids.add(node.dataset.node);
            send(node, 'select', {ids: Array.from(ids)});
        }
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
