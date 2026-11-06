<script setup lang="ts">
withDefaults(defineProps<{ size?: number }>(), { size: 50 })
</script>

<template>
  <svg :width="size" :height="size" viewBox="0 0 64 64" role="img" aria-label="Synodos logo"
    class="synodos-mark" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="synodos-speech-gradient" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#8850c9" />
        <stop offset="1" stop-color="#4c1d83" />
      </linearGradient>
    </defs>
    <path d="M15.5 15.5c-9 3.5-12.4 12-9.2 20.1 1.3 3.5 3.9 6 7.4 7.8l-3 9.8 12.9-6c6.3 1.1 14 .2 20-4.1 8.5-5.9 10.9-16.8 4.5-24.1-6.6-7.4-21.7-9.1-32.6-3.5Z"
      fill="url(#synodos-speech-gradient)" />
    <path d="M29 8.6c11.1-5.2 25-1.7 29 10.1 4.9 13.7-3.9 28.1-17.8 31.5-4.2 1-8.9 1-13.1-.2"
      fill="none" stroke="#FD621B" stroke-width="4.4" stroke-linecap="round" />
    <path d="M18.5 23h24M22 27h17M23.4 39.5h13.7M27 27v12M32 27v12M37 27v12"
      fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
    <path d="M31 18.9h.01" stroke="#fff" stroke-width="3.2" stroke-linecap="round" />
  </svg>
</template>
