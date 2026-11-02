import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('@/pages/HomePage.vue'),
    },
    {
      path: '/meeting/:id',
      name: 'meeting',
      component: () => import('@/pages/MeetingPage.vue'),
      props: true,
    },
    {
      path: '/speakers',
      name: 'speakers',
      component: () => import('@/pages/SpeakersPage.vue'),
    },
    {
      path: '/speakers/new',
      name: 'speaker-new',
      component: () => import('@/pages/SpeakerNewPage.vue'),
    },
    {
      path: '/hotwords',
      name: 'hotwords',
      component: () => import('@/pages/HotwordsPage.vue'),
    },
    {
      path: '/hotwords/new',
      name: 'hotword-new',
      component: () => import('@/pages/HotwordNewPage.vue'),
    },
  ],
})

export default router
