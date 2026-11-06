<script setup lang="ts">
withDefaults(defineProps<{ size?: number }>(), { size: 50 })
</script>

<template>
  <svg :width="size" :height="size" viewBox="0 0 64 64" role="img"
    aria-label="Synodos logo" class="synodos-mark" xmlns="http://www.w3.org/2000/svg">
    <path
      d="M19 8h26c8 0 14 6.1 14 14v16c0 7.7-6 13.8-14 13.8H25L12 60l2.2-11.8C8.5 46.3 5 41.5 5 35.5V22C5 14.1 11.1 8 19 8Z"
      fill="#EB7100"
    />
    <path d="M20 21.5h24M23 27h18M26 31v10M32 31v10M38 31v10M22 44.5h20"
      stroke="#fff" stroke-width="3.25" stroke-linecap="round" stroke-linejoin="round" fill="none" />
    <path d="M20 21.5h24" stroke="#292521" stroke-width="2.5" stroke-linecap="round" />
  </svg>
</template>
