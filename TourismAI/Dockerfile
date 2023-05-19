FROM rust:1.68 as build
WORKDIR /tourism_ai

COPY ./src ./src
COPY ./Cargo.lock ./Cargo.lock
COPY ./Cargo.toml ./Cargo.toml

RUN cargo install --path .

FROM debian:stable-slim

COPY ./Data ./Data
COPY --from=build /usr/local/cargo/bin/tourism_ai /usr/local/bin/tourism_ai

CMD tourism_ai