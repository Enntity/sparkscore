# SPDX-License-Identifier: AGPL-3.0-only
# Context: build.sh's pinned Atlas checkout (engine/), install/ and tools/.
# The builder stage follows SparkQwen's image (same CUDA, CUTLASS and Rust
# pins); the target is gb10 / qwen3.8-27b / nvfp4, the dense 27B kernels.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
    git ca-certificates build-essential python3 curl pkg-config cmake libclang-dev \
    libibverbs-dev librdmacm-dev libnccl2=2.31.2-1+cuda13.3 libnccl-dev=2.31.2-1+cuda13.3 && \
    rm -rf /var/lib/apt/lists/*
# rustup-init at a pinned release, checked against its published SHA-256
# before it runs (the image is arm64 only).
ARG RUSTUP_VERSION=1.29.1
ARG RUSTUP_SHA256=15f6e4ce9f583b929c996c91562bad6d4454f3281de858b02cdfdef615fac433
RUN curl --proto '=https' --tlsv1.2 -sSfLo /tmp/rustup-init \
      "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/aarch64-unknown-linux-gnu/rustup-init" && \
    echo "${RUSTUP_SHA256}  /tmp/rustup-init" | sha256sum -c - && chmod +x /tmp/rustup-init && \
    /tmp/rustup-init -y --profile minimal --default-toolchain 1.93.1 && rm /tmp/rustup-init
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_TOOLCHAIN=1.93.1 CUDA_HOME=/usr/local/cuda CUTLASS_HOME=/opt/cutlass
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
RUN git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass && \
    git -C /opt/cutlass checkout cf064d2e6bad2886238ac565b3b49007764f4939
WORKDIR /build
COPY engine/ /build/
ENV ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4
ARG BUILD_JOBS=8
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target-sparkscore,sharing=shared \
    cargo build --locked --release -j "$BUILD_JOBS" -p spark-server && \
    install -D target/release/spark /build/dist/spark
# License notices of every third-party crate compiled into the binary.
COPY install/rust-notices.py /tmp/rust-notices.py
RUN python3 /tmp/rust-notices.py /build /build/dist/notices-rust

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
    python3 python3-venv ca-certificates curl libnccl2=2.31.2-1+cuda13.3 libibverbs1 librdmacm1 \
    ibverbs-providers libnl-3-200 libnl-route-3-200 && rm -rf /var/lib/apt/lists/*
# The harness needs a tokenizer and the chat template only: no torch.
COPY tools/sparkscore/requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -q -r /tmp/requirements.txt
COPY --from=builder /build/dist/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY --from=builder /opt/cutlass/LICENSE.txt /opt/sparkscore/notices/CUTLASS-LICENSE.txt
COPY --from=builder /build/dist/notices-rust/ /opt/sparkscore/notices/rust/
COPY --from=builder /build/LICENSE /opt/sparkscore/notices/ATLAS-LICENSE
COPY tools/ /opt/sparkscore/tools/
COPY install/serve.py install/checkpoints.json /opt/sparkscore/
COPY install/atlas-source.json /opt/sparkscore/atlas-source.json
COPY LICENSE NOTICE /opt/sparkscore/
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV RUST_LOG=info PYTHONDONTWRITEBYTECODE=1
EXPOSE 8015
HEALTHCHECK --interval=10s --timeout=3s --start-period=600s CMD curl -sf http://127.0.0.1:8015/health || exit 1
# The image's identity is the git trees of install/ and tools/ (its tag).
ARG SOURCE_TREE
RUN test -n "$SOURCE_TREE"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.source="https://github.com/Enntity/sparkscore" \
      io.enntity.sparkscore.source-tree="$SOURCE_TREE"
WORKDIR /opt/sparkscore
ENTRYPOINT ["/opt/venv/bin/python", "/opt/sparkscore/serve.py"]
