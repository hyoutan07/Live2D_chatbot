<template>
  <div id="live2d-box">
  </div>
</template>

<script lang="ts">
import { dummy } from '../static/bundle'

export default {
  asyncData() {
    dummy()
  }
}
</script>

<style>
canvas {
  /* position: absolute;
  top: 0;
  left: 0; */
}
</style>