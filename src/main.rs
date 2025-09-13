use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use egui_plot::{Line, Plot, PlotPoints};
use std::{
    collections::VecDeque,
    fs::File,
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

// Interfaz
// Tenemos dos contextos concurrentes:
// 1. El callback de audio (lo llama la librería en un hilo propio de alta
//    frecuencia)
// 2. El hilo de la interfaz gráfica (dibuja y responde a clicks)
// Ambos necesitan leer y escribir parte del estado de la estructura de datos
// que vamos a construir: Shared agrupa esos datos y se pasa a ambos lados a
// través de un Arc<Shared>
struct Shared {
    // Es un flag leído por el callback para decidir si acumular muestras.
    // - Elegimos AtomicBool para evitar un Mutex sólo por un booleano.
    // - Lecturas/escrituras son lock-free y baratas.
    // - Usamos Ordering::SeqCst en el código por simplicidad y seguridad;
    //   con más análisis se podría usar Ordering::Relaxed para este caso
    //   (siempre que no dependas de ordenamientos con otros datos),
    //   pero SeqCst evita sorpresas.
    is_recording: AtomicBool,

    // Ring buffer para visualización (muestras normalizadas -1..1 del primer
    // canal):
    // - El visualizador necesita un ring buffer de amplitudes recientes.
    // - VecDeque permite eliminar por delante (pop_front) cuando alcanzamos
    //   capacidad.
    // - Protegido con Mutex porque lo escribe el callback y lo lee la GUI.
    // - Nota de ingeniería de audio: en tiempo real idealmente evitaríamos
    //   Mutex en el callback (puede introducir XRUNs si el bloqueo se
    //   prolonga). Para una app sencilla está bien; podríamos cambiar a una
    //   cola lock-free (por ejemplo, rtrb, crossbeam::ArrayQueue o un ring
    //   buffer atómico propio).
    vis_buffer: Mutex<VecDeque<f32>>,

    // Muestras de audio guardadas (mono, i16)
    // - Aquí acumulamos las muestras que sí vamos a guardar en el WAV.
    // - Elegimos i16 porque el WAV final es PCM 16-bit; así evitamos convertir
    //   al guardar.
    // - Se limpia antes de empezar a grabar; al guardar detenemos la grabación
    //   para evitar contención y una carrera entre guardar y seguir
    //   escribiendo.
    recorded: Mutex<Vec<i16>>,
}

// Constructor
// Permitimos parametrizar la capacidad del visualizador (vis_capacity), porque
// el tamaño del ring buffer afecta:
// - Memoria y coste de dibujo.
// - La "ventana temporal" visible del waveform.
impl Shared {
    fn new(vis_capacity: usize) -> Self {
        Self {
            is_recording: AtomicBool::new(false),
            vis_buffer: Mutex::new(VecDeque::with_capacity(vis_capacity)),
            recorded: Mutex::new(Vec::new()),
        }
    }
}
// Representa un item de la lista de entradas de audio que mostramos en el
// combo de la GUI.
struct InputDev {
    // Una etiqueta de usuario para mostrar en la GUI. Nos evita repetir llama-
    // das a device.name() -> Result(String, _)
    name: String,
    // el handle real de cpal que usamos para consultar configuraciones y
    // crear el stream. Guardarlo por valor nos permite:
    // - Consultar sus formatos (default_input_config, supported_input_configs)
    // - Construir el stream sin préstamos a estructuras externas
    // - Liberar el recurso cuando cambiamos de dispositivo o al cerrar
    device: cpal::Device,
}

// Bastante descriptivo: representa el estado de la GUI
struct AppState {
    // Estado compartido entre el callback de audio (thread cpal) y la GUI
    // Arc permite compartir sin copiar. Mutex y Atomic protegen datos.
    shared: Arc<Shared>,

    // Catálogo de entradas detectadas para poblar el combo de selección
    // Se refresca bajo demanda (click del usuario)
    input_devs: Vec<InputDev>,

    // Índice del dispositivo elegido en input_devs. Mantener el índice, no un
    // préstamo, evita problemas de borrow.
    selected_input: usize,

    // El stream activo de captura. Usamos Option porque:
    // - Some(stream) cuando estamos grabando.
    // - None cuando no.
    // Soltarlo (None) libera el dispositivo inmediatamente.
    stream: Option<cpal::Stream>,

    // Configuración efectiva que usa el stream (no preferencias, sino lo que
    // conseguimos abrir). Se muestran en la UI y gobiernan el guardado.
    sample_rate: u32,
    channels: u16,

    // Marca para el temporizador. None cuando no hay grabación; Some(t0) para
    // calcular el tiempo transcurrido.
    started_at: Option<Instant>,
    // Equilibrio entre coste y fluidez: actualizamos el waveform cada ~30 ms,
    // no en cada frame.
    last_vis_pull: Instant,
    // Copia "estable" de las muestras visibles para dibujar. Evita dibujar
    // directamente sobre el Mutex del ring buffer.
    vis_points: Vec<f32>,
}

impl AppState {
    // El constructor: para un muggle como yo, esto es auténtica
    // brujería, pero intentaré darle una explicación
    fn new() -> Self {
        // guardamos el host por defecto
        let host = cpal::default_host();
        // inicializamos un vector para almacenar el listado de dispositivos
        let mut input_devs = Vec::new();
        // Insertamos los dispositivos en nuestro vector
        if let Ok(devices) = host.input_devices() {
            for d in devices {
                let name = d
                    .name()
                    .unwrap_or_else(|_| "Dispositivo sin nombre".to_string());
                input_devs.push(InputDev { name, device: d });
            }
        }
        // Si no hay dispositivos, intentamos al menos añadir el dispositivo por
        // defecto del host.
        if input_devs.is_empty() {
            if let Some(d) = host.default_input_device() {
                let name = d
                    .name()
                    .unwrap_or_else(|_| "Entrada por defecto".to_string());
                input_devs.push(InputDev { name, device: d });
            }
        }
        // Calculamos una configuración inicial (sample_rate, channels) a partir
        // del primer dispositivo disponible, con default_config_for.
        let (sample_rate, channels) = if let Some(dev) = input_devs.get(0) {
            default_config_for(&dev.device).unwrap_or((44_100, 1))
        } else {
            (44_100, 1)
        };

        // Crea el estado inicial: Shared con ring buffer de 4096 muestras,
        // selected_input = 0, sin stream activo, etc
        Self {
            shared: Arc::new(Shared::new(4096)),
            input_devs,
            selected_input: 0,
            stream: None,
            sample_rate,
            channels,
            started_at: None,
            last_vis_pull: Instant::now(),
            vis_points: Vec::new(),
        }
    }

    fn start_recording(&mut self) {
        if self.input_devs.is_empty() || self.stream.is_some() {
            return;
        }
        let dev = &self.input_devs[self.selected_input].device;
        let (sr, ch) = default_config_for(dev).unwrap_or((self.sample_rate, self.channels));
        self.sample_rate = sr;
        self.channels = ch;

        let config = cpal::StreamConfig {
            channels: ch,
            sample_rate: cpal::SampleRate(sr),
            buffer_size: cpal::BufferSize::Default,
        };

        // Limpia buffers previos
        {
            let mut rec = self.shared.recorded.lock().unwrap();
            rec.clear();
            let mut vis = self.shared.vis_buffer.lock().unwrap();
            vis.clear();
        }
        self.shared.is_recording.store(true, Ordering::SeqCst);

        let err_fn = |e| eprintln!("Error en stream: {e}");
        let shared = self.shared.clone();
        let ch_usize = ch as usize;

        let stream = match dev.default_input_config() {
            Ok(def_cfg) => {
                use cpal::SampleFormat::*;
                match def_cfg.sample_format() {
                    F32 => dev.build_input_stream(
                        &config,
                        move |data: &[f32], _| {
                            // Extrae primer canal
                            let first: Vec<f32> =
                                data.chunks(ch_usize).map(|frame| frame[0]).collect();
                            on_input_data_any(&shared, &first);
                        },
                        err_fn,
                        None,
                    ),
                    I16 => dev.build_input_stream(
                        &config,
                        move |data: &[i16], _| {
                            let first: Vec<f32> = data
                                .chunks(ch_usize)
                                .map(|frame| frame[0] as f32 / i16::MAX as f32)
                                .collect();
                            on_input_data_any(&shared, &first);
                        },
                        err_fn,
                        None,
                    ),
                    U16 => dev.build_input_stream(
                        &config,
                        move |data: &[u16], _| {
                            let first: Vec<f32> = data
                                .chunks(ch_usize)
                                .map(|frame| (frame[0] as f32 / u16::MAX as f32) * 2.0 - 1.0)
                                .collect();
                            on_input_data_any(&shared, &first);
                        },
                        err_fn,
                        None,
                    ),
                    _ => {
                        eprintln!("Formato de muestra no soportado por este ejemplo.");
                        return;
                    }
                }
            }
            Err(e) => {
                eprintln!("No se pudo obtener configuración por defecto: {e}");
                return;
            }
        }
        .ok();

        if let Some(s) = stream {
            if s.play().is_ok() {
                self.stream = Some(s);
                self.started_at = Some(Instant::now());
            }
        }
    }

    fn stop_recording(&mut self) {
        self.shared.is_recording.store(false, Ordering::SeqCst);
        self.stream = None;
        self.started_at = None;
    }

    fn save_as_wav(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .set_title("Guardar como WAV")
            .add_filter("WAV", &["wav"])
            .save_file()
        else {
            return;
        };

        // Pausa grabación mientras guardas (evita contendencias con el callback)
        let was_recording = self.stream.is_some();
        if was_recording {
            self.stop_recording();
        }

        // Ajusta el tipo si tu buffer es f32; en mi versión es Vec<i16>
        let samples: Vec<i16> = {
            let lock = self.shared.recorded.lock().unwrap();
            lock.clone()
        };

        if samples.is_empty() {
            eprintln!("No hay audio grabado para guardar.");
            if was_recording {
                self.start_recording();
            }
            return;
        }

        let spec = hound::WavSpec {
            channels: 1, // mono (primer canal)
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        // Guardado robusto: finalize() + fsync con un clon del File
        let res = (|| -> Result<(), Box<dyn std::error::Error>> {
            // Creamos el archivo y un clon del descriptor para fsync posterior
            let file = File::create(&path)?;
            let mut file_for_sync = file.try_clone()?;

            // Escribimos vía BufWriter + WavWriter
            let bufw = BufWriter::new(file);
            let mut wav = hound::WavWriter::new(bufw, spec)?;

            for &s in &samples {
                wav.write_sample(s)?;
            }

            // Cierra cabecera/tamaños y suelta el WavWriter (cierra el writer subyacente)
            wav.finalize()?;

            // Fuerza escritura al disco en el clon (mismo fichero)
            file_for_sync.flush()?;
            file_for_sync.sync_all()?;
            // Al salir del bloque, ambos File se dropean y se cierran.

            Ok(())
        })();

        if let Err(e) = res {
            eprintln!("Error al guardar WAV: {e}");
        }

        if was_recording {
            self.start_recording();
        }
    }
}

fn default_config_for(dev: &cpal::Device) -> Option<(u32, u16)> {
    let cfg = dev.default_input_config().ok()?;
    Some((cfg.sample_rate().0, cfg.channels()))
}

/// Recibe muestras normalizadas (-1..1) del primer canal y:
/// 1) Acumula en `recorded` como i16 mono
/// 2) Alimenta el ring buffer de visualización
fn on_input_data_any(shared: &Arc<Shared>, first_channel: &[f32]) {
    if !shared.is_recording.load(Ordering::SeqCst) {
        return;
    }

    // Guardar WAV (i16)
    {
        let mut rec = shared.recorded.lock().unwrap();
        rec.reserve(first_channel.len());
        for &s in first_channel {
            let clamped = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            rec.push(clamped);
        }
    }

    // Visualización
    {
        let mut vis = shared.vis_buffer.lock().unwrap();
        for &s in first_channel {
            if vis.len() >= vis.capacity() {
                let _ = vis.pop_front();
            }
            vis.push_back(s.clamp(-1.0, 1.0));
        }
    }
}

impl eframe::App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.heading("Mic Recorder");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Entrada:");
                if !self.input_devs.is_empty() {
                    // Selector de dispositivo sin violar el borrow checker
                    let mut pending = self.selected_input;
                    egui::ComboBox::from_id_source("input_sel")
                        .selected_text(
                            self.input_devs
                                .get(self.selected_input)
                                .map(|d| d.name.as_str())
                                .unwrap_or("Entrada por defecto"),
                        )
                        .show_ui(ui, |ui| {
                            for i in 0..self.input_devs.len() {
                                let name = self.input_devs[i].name.clone();
                                ui.selectable_value(&mut pending, i, name);
                            }
                        });
                    if pending != self.selected_input {
                        self.selected_input = pending;
                        if self.stream.is_some() {
                            self.stop_recording();
                            self.start_recording();
                        }
                    }
                } else {
                    ui.colored_label(egui::Color32::RED, "No hay dispositivos de entrada");
                }

                if self.stream.is_none() {
                    if ui.button("Grabar").clicked() {
                        self.start_recording();
                    }
                } else {
                    if ui.button("Parar").clicked() {
                        self.stop_recording();
                    }
                }

                if ui.button("💾 Guardar como…").clicked() {
                    self.save_as_wav();
                }
            });

            ui.separator();

            // Timer e info
            let elapsed = if let Some(t0) = self.started_at {
                let d = t0.elapsed();
                format!(
                    "{:02}:{:02}.{:03}",
                    d.as_secs() / 60,
                    d.as_secs() % 60,
                    d.subsec_millis()
                )
            } else {
                "00:00.000".to_string()
            };
            let n_samples = self.shared.recorded.lock().unwrap().len();
            ui.label(format!(
                "Tiempo: {elapsed}   |   {} Hz, {} ch (entrada)   |  Muestras: {}",
                self.sample_rate, self.channels, n_samples
            ));

            ui.separator();

            // Actualiza caché de visualización ~33 FPS
            if self.last_vis_pull.elapsed() >= Duration::from_millis(30) {
                let lock = self.shared.vis_buffer.lock().unwrap();
                self.vis_points.clear();
                self.vis_points.extend(lock.iter().copied());
                self.last_vis_pull = Instant::now();
            }

            let points: PlotPoints = self
                .vis_points
                .iter()
                .enumerate()
                .map(|(i, &y)| [i as f64, y as f64])
                .collect();

            Plot::new("waveform")
                .height(200.0)
                .allow_scroll(false)
                .allow_zoom(false)
                .allow_drag(false)
                .allow_boxed_zoom(false)
                .show_axes([false, false]) // oculta ejes X e Y
                .show_grid(false) // oculta rejilla
                // y fijo en amplitud:
                .include_y(-1.2)
                .include_y(1.2)
                // rango X acorde al buffer:
                .include_x(0.0)
                .include_x(self.vis_points.len().max(1) as f64)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(points));
                });
            ui.add_space(6.0);
        });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        ..Default::default()
    };
    eframe::run_native(
        "Mic Recorder",
        native_options,
        Box::new(|_| Box::new(AppState::new())),
    )
}
