// 3D backdrop: rotating network globe + drifting particles (Three.js), and card tilt effect.
(function () {
    const canvas = document.getElementById('scene3d');
    if (window.THREE && canvas) {
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
        camera.position.z = 9;

        const globe = new THREE.Group();
        scene.add(globe);

        // Wireframe sphere
        const R = 3;
        const wire = new THREE.LineSegments(
            new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(R, 3)),
            new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.16 })
        );
        globe.add(wire);

        // Nodes on the sphere surface + links between nearby nodes
        const nodes = [];
        const N = 90;
        for (let i = 0; i < N; i++) {
            const phi = Math.acos(1 - 2 * (i + 0.5) / N);
            const theta = Math.PI * (1 + Math.sqrt(5)) * i;
            nodes.push(new THREE.Vector3(
                R * Math.sin(phi) * Math.cos(theta), R * Math.cos(phi), R * Math.sin(phi) * Math.sin(theta)));
        }
        const pts = new THREE.BufferGeometry().setFromPoints(nodes);
        const nodeMat = new THREE.PointsMaterial({ color: 0x67e8f9, size: 0.09, transparent: true, opacity: 0.95 });
        globe.add(new THREE.Points(pts, nodeMat));

        const linkPts = [];
        for (let i = 0; i < N; i++)
            for (let j = i + 1; j < N; j++)
                if (nodes[i].distanceTo(nodes[j]) < 1.1) linkPts.push(nodes[i], nodes[j]);
        globe.add(new THREE.LineSegments(
            new THREE.BufferGeometry().setFromPoints(linkPts),
            new THREE.LineBasicMaterial({ color: 0x818cf8, transparent: true, opacity: 0.5 })
        ));

        // Inner glowing core
        const core = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1.2, 1),
            new THREE.MeshBasicMaterial({ color: 0x6366f1, wireframe: true, transparent: true, opacity: 0.35 })
        );
        globe.add(core);

        // Orbit rings
        for (let k = 0; k < 3; k++) {
            const ring = new THREE.Mesh(
                new THREE.TorusGeometry(R + 0.6 + k * 0.35, 0.008, 8, 160),
                new THREE.MeshBasicMaterial({ color: k === 1 ? 0xa78bfa : 0x22d3ee, transparent: true, opacity: 0.35 })
            );
            ring.rotation.x = Math.PI / 2 + k * 0.5;
            ring.rotation.y = k * 0.6;
            globe.add(ring);
        }

        // "Threat" markers pulsing red
        const threats = [];
        for (let i = 0; i < 6; i++) {
            const m = new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 12),
                new THREE.MeshBasicMaterial({ color: 0xfb7185, transparent: true }));
            m.position.copy(nodes[(i * 13 + 5) % N]);
            globe.add(m);
            threats.push(m);
        }

        // Background particles
        const pGeo = new THREE.BufferGeometry();
        const pArr = new Float32Array(600 * 3);
        for (let i = 0; i < pArr.length; i++) pArr[i] = (Math.random() - 0.5) * 40;
        pGeo.setAttribute('position', new THREE.BufferAttribute(pArr, 3));
        const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({ color: 0x94a3b8, size: 0.05, transparent: true, opacity: 0.5 }));
        scene.add(particles);

        let mx = 0, my = 0;
        window.addEventListener('mousemove', e => {
            mx = (e.clientX / window.innerWidth - 0.5);
            my = (e.clientY / window.innerHeight - 0.5);
        });

        function resize() {
            const w = window.innerWidth, h = window.innerHeight;
            renderer.setSize(w, h, false);
            camera.aspect = w / h;
            // push the globe to the right on wide screens, centre on narrow ones
            globe.position.x = w > 900 ? 4.2 : 0;
            globe.position.y = w > 900 ? 0 : 3;
            camera.updateProjectionMatrix();
        }
        window.addEventListener('resize', resize);
        resize();

        const clock = new THREE.Clock();
        (function loop() {
            const t = clock.getElapsedTime();
            globe.rotation.y = t * 0.12 + mx * 0.6;
            globe.rotation.x = 0.25 + my * 0.4;
            core.rotation.y = -t * 0.4;
            core.rotation.x = t * 0.25;
            threats.forEach((m, i) => {
                const s = 1 + 0.9 * Math.abs(Math.sin(t * 2 + i));
                m.scale.setScalar(s);
                m.material.opacity = 1 - 0.55 * Math.abs(Math.sin(t * 2 + i));
            });
            particles.rotation.y = t * 0.01;
            renderer.render(scene, camera);
            requestAnimationFrame(loop);
        })();
    }

    // 3D tilt for cards
    document.addEventListener('mousemove', e => {
        document.querySelectorAll('.tilt').forEach(el => {
            const r = el.getBoundingClientRect();
            if (e.clientX < r.left - 40 || e.clientX > r.right + 40 || e.clientY < r.top - 40 || e.clientY > r.bottom + 40) {
                el.style.transform = ''; return;
            }
            const x = (e.clientX - r.left) / r.width - 0.5;
            const y = (e.clientY - r.top) / r.height - 0.5;
            el.style.transform = `perspective(900px) rotateX(${(-y * 8).toFixed(2)}deg) rotateY(${(x * 10).toFixed(2)}deg) translateZ(6px)`;
        });
    });
})();
