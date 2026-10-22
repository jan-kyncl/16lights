# Statický frontend za Caddy (build context = frontend/).
FROM node:22-alpine AS build
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM caddy:2.8-alpine
COPY --from=build /src/dist /srv
