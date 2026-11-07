<script setup lang="ts">
import { onMounted, ref } from 'vue'
import AppIcon from '@/components/AppIcon.vue'

type Health = {
  status?: string
  realtime?: { asr_engine?: string; asr_model?: string; independent_diarization?: string }
  active_meetings?: number
}
const loading = ref(true)
const health = ref<Health | null>(null)
const error = ref('')
async function refreshHealth(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    const response = await fetch('/api/health')
    if (!response.ok) throw new Error('Backend returned HTTP ' + response.status)
    health.value = (await response.json()) as Health
  } catch (reason) {
    health.value = null
    error.value = reason instanceof Error ? reason.message : String(reason)
  } finally {
    loading.value = false
  }
}
onMounted(() => { void refreshHealth() })
</script>

<template>
  <div class="synodos-settings-page">
    <header class="synodos-settings-header">
      <span class="synodos-kicker">WORKSPACE PREFERENCES</span>
      <h1>Settings</h1>
      <p>See the runtime connected to this workspace. Meeting-specific capture options live inside each meeting.</p>
    </header>
    <section class="synodos-settings-grid">
      <article class="synodos-settings-card">
        <div class="synodos-settings-card-icon"><AppIcon name="mic" :size="22" /></div>
        <h2>Speech engine</h2>
        <p v-if="loading">Checking backend connection...</p>
        <p v-else-if="error" role="status">Backend unavailable: {{ error }}</p>
        <template v-else-if="health">
          <dl>
            <div><dt>Status</dt><dd>{{ health.status === 'ok' ? 'Connected' : health.status || 'Unknown' }}</dd></div>
            <div><dt>Recognition</dt><dd>{{ health.realtime?.asr_engine || 'Not reported' }}</dd></div>
            <div><dt>Model</dt><dd>{{ health.realtime?.asr_model || 'Not reported' }}</dd></div>
            <div><dt>Diarization</dt><dd>{{ health.realtime?.independent_diarization || 'Off' }}</dd></div>
          </dl>
        </template>
        <button class="btn btn-ghost" type="button" :disabled="loading" @click="refreshHealth">
          <AppIcon name="list" :size="16" /> Refresh status
        </button>
      </article>
      <article class="synodos-settings-card">
        <div class="synodos-settings-card-icon"><AppIcon name="settings" :size="22" /></div>
        <h2>Meeting controls</h2>
        <p>Choose the recognition language, translation output, saved speakers, microphone thresholds and recording options in an individual meeting.</p>
        <RouterLink class="btn btn-primary" to="/#meetings">
          <AppIcon name="file-text" :size="16" /> Open meetings
        </RouterLink>
      </article>
      <article class="synodos-settings-card">
        <div class="synodos-settings-card-icon"><AppIcon name="check" :size="22" /></div>
        <h2>Local-first workspace</h2>
        <p>Synodos runs with your configured local services. Audio, model selection and stored meeting information are controlled by your backend configuration.</p>
      </article>
    </section>
  </div>
</template>
