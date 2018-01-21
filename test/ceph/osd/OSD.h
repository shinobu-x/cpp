class OSDService
public:
  OSD* osd;
  CephContext* cct;
  SharedPtrRegistry<spg_t, ObjectStore::Sequence> osr_registry;
  ceph::shared_ptr<ObjectStore::Sequencer> meta_osr;
  const int whoami;
  ObjectStore*& store;
  LogClient& log_clinet;
  LogChannelRef clog;
  PGRecoveryStats& pg_recovery_stats;

  PerfCounters*& logger;
  PerfCounters*& recoverystate_perf;
  MonClient*& monc;
  GenContextWQ recovery_gen_wq;
  ClassHandler*& class_handler;

  void enqueue_back(OpQueueItem&&);
  void enqueue_front(OpQueueItem&&);
  void maybe_inject_dispatch_delay();

  void pg_add_epoch(spg_t, epoch_t);
  void pg_update_epoch(spg_t, epoch_t);
  void pg_remove_epoch(spg_t);
  epoch_t get_min_pg_epoch();
  void wait_min_pg_epoch(epoch_t);

  OSDSuperblock get_superblock();
  void publish_superblock(const OSDSuperblock& block);
  int get_nodeid() const;
  std::atomic<epoch_t> max_oldest_map;

  OSDMapRef get_osdmap();
  epoch_t get_osdmap_epoch();
  void publish_map(OSDMapRef map);

  void pre_publish_map(OSDMapRef map);
  void activate_map();
  // map epochs reserved below
  map<epoch_t, unsigned> map_reservation;
  // gets ref to next_osdmap and registers the epoch as reserved
  OSDMapRef get_nextmap_reserved();
  // releases reservation on map
  void release_map(OSDMapRef osdmap);
  // blocks until there are no reserved maps prior to next_osdmap
  void await_reserved_maps();
  OSDMapRef get_next_osdmap();

  epoch_t get_peer_epoch(int);
  epoch_t note_peer_epoch(int, epoch_t);
  void forget_peer_epoch(int, epoch_t);

  void send_map(class MOSDMap*, Connection*);
  void send_incremental_map(epoch_t, Connection*, OSDMapRef&);
  MOSDMap* build_incremental_map_msg(epoch_t, epoch_t, OSDSuperblock&);
  bool should_share_map(entity_name_t, Connection, epoch_t);
  void share_map(entity_name_t, Connection, epoch_t, OSDMapRef&, epoch_t*);
  void share_map_peer(int, Connection*, OSDMapRef);
  ConnectionRef get_con_osd_cluster(int, epoch_t);
  pair<ConnectionRef, ConnectionRef> get_con_osd_hb(int, epoch_t);
  void send_message_osd_cluster(int, Message, epoch_t);
  void send_message_osd_cluster(Message, Connection);
  void send_message_osd_cluster(Message, const ConnectionRef&);
  void send_message_osd_client(Message, Connection);
  void send_message_osd_client(Message, const ConnectionRef&);
  entity_name_t get_cluster_msgr_name();

/* ScrubJob */
  struct ScrubJob
    CephContext* cct;
    // pg to be scrubbed
    spg_t pgid;
    // a time scheduled for scrub. but the scrub could be delayed if system
    // load is too high or it fails to fall in the scrub hours
    utime_t sched_time;
    // the hard upper bound of scrub time
    utime_t deadline;
    explicit ScrubJob(CephContext*, const spg_t&, const utime_t&,
      double, double, bool);
    // order the jobs by sched_time
    bool operator<(const ScrubJob&) const;
/* ScrubJob */

  set<ScrubJob> sched_scrub_pg;
  // @returns the scrub_reg_stamp used for unregister the scrub job
  utime_t reg_pg-scrub(spg_t, utime_t, double, double, bool);
  void unreg_pg_scrub(spg_t, utime_t);
  bool first_scrub_stamp(ScrubJob* out);
  bool next_scrub_stamp(const ScrubJob&, ScrubJob* out);
  void dumps_scrub(Formatter* f);
  bool can_inc_scrubs_pending();
  bool inc_scrubs_pending();
  void inc_scrubs_active(bool);
  void dec_scrubs_pending();
  void dec_scrubs_active();
  void reply_op_error(OpRequestRef, int);
  void reply_op_error(OpRequestRef, int, eversion_t, version_t);
  void handle_misdirected_op(PG*, OpRequestRef);

  void agent_entry();
  void agent_stop();
  void _enqueque(PG*, uint64_t);
  void _dequeue(PG*, uint64_t);
  // enable agent for a pg
  void agent_enable_pg(PG*, uint64_t);
  // adjust priority for an enabled pg
  void agent_adjust_pg(PG*, uint64_t, uint64_t);
  // disable agnet for a pg
  void agent_disable_pg(PG*, uint64_t);
  // note start of an async (evict) op
  void agent_start_evict_opp();
  // note finish or cancellation of async (evict) op
  void agent_finish_evict_op();
  // note start of an async (flush) op
  void agent_start_op(const hobject_t&);
  // note finish or cancellation of an async (flush) op
  void agent_finish_op(const hobject_t&);
  // check if we are operating on an object
  bool agent_is_active_oid(const hobject_t&);
  // get count of active agent ops
  int agent_get_num_ops();
  void agent_inc_high_count();
  void agent_dec_high_count();

  bool promote_throttle();
  void promote_finish(uint64_t);
  void promote_throttle_recalibrate();
  // Objecter, for tiering reads/writes from/to other OSDs
  Objecter* objecter;
  int m_objecter_finishers;
  vector<Finisher*> objecter_finishers;
  Mutex watch_lock;
  SafeTimer watch_timer;
  uint64_t next_notif_id;
  uint64_t get_next_id(epoch_t);
  Mutex recovery_request_lock;
  SafeTimer recovery_request_timer;
  // For async recovery sleep
  bool recovery_needs_sleep;
  utime_t recovery_schedule_time;
  Mutex recovery_sleep_lock;
  SafeTimer recovery_sleep_timer;
  // for ops i issue
  std::atomic<unsigned int> last_tid{0};
  ceph_tid_t get_tid();

  Finisher reserver_finisher;
  AsyncRecovery<spg_t> local_reserver;
  AsyncRecovery<spg_t> remote_reserver;

  void queue_want_pg_temp(pg_t pgid, const vector<int>&);
  void remove_want_pg_temp(pg_t);
  void requeue_pg_temp();
  void send_pg_temp();
  void send_pg_created(pg_t);
  Mutex snap_sleep_lock;
  SafeTimer snap_sleep_timer;
  Mutex scrub_sleep_lock;
  SafeTimer scrub_sleep_timer;
  AsyncReserver<spg_t> snap_reserver;
  void queue_for_snap_trim(PG*);
  void queue_for_scrub(PG*, bool);
  void queue_for_pg_delete(spg_t, epoch_t);
  void finish_pg_delete(PG*);

  void start_recovery_op(PG*, const hobject_t&);
  void finish_recovery_op(PG*, const hobject_t&, bool);
  bool is_recovery_active();
  void release_reserved_pushes(uint64_t);
  void release_reserved_pushes(uint64_t);
  void defer_recovery(float);
  void defer_recovery();
  bool recovery_is_paused();
  void unpause_recovery();
  void kick_recovery_queue();
  void clear_queued_recovery(PG*);
  // delayed pg activation
  void queue_for_recovery(PG*);
  void queue_recovery_sleep(PG*, epoch_t, uint64_t);
  // osd map cache (past osd maps)
  Mutex map_cache_lock;
  SharedLRU<epoch_t, const OSDMap> map_cache;
  SharedLRU<epoch_t, bufferlist> map_bl_cache;
  SharedLRU<epoch_t, bufferlist> map_bl_inc_cache;
  // newest map fully consumed by handle_osd_map (i.e., written to disk)
  epoch_t map_cache_pinned_epoch = 0;
  // true if pg consumption affected our unpinning
  std::atomic<bool> map_cache_pinned_low;
  map<int64_t, int> deleted_pool_pg_nums;
  OSDMapRef try_get_map(epoch_t e);
  OSDMapRef get_map(epoch_t);
  OSDMapRef add_map(OSDMap*);
  OSDMapRef _add_map(OSDMap*);
  void add_map_bl(epoch_t, bufferlist&);
  void pin_map_bl(epoch_t, bufferlist&);
  void _add_map_bl(epoch_t e, bufferlist&);
  bool get_map_bl(epoch_t e, bufferlist&);
  bool _get_map_bl(epoch_t, bufferlist&);
  void add_map_inc_bl(epoch_t, bufferlist&);
  void pin_map_inc_bl(epoch_t, bufferlist&);
  void _add_map_inc_bl(epoch_t, bufferlist&);
  bool get_inc_map_bl(epoch_t, bufferlist&);
  void clear_map_bl_cache_pins(epoch_t);
  void check_map_bl_cache_pins();
  int get_deleted_pool_pg_num(uint64_t);
  void store_deleted_pool_pg-num(int64_t, int);
  int get_possibly_deleted_pool_pg_num(OSDMapRef, int64_t);
  void need_heartbeat_peer_update();
  void init();
  void final_init();
  void start_shutdown();
  void shutdown_resever();
  void shutdown();

  void _start_split(spg_t, const set<spg_t>);
  void start_aplit(spg_t, const set<spg_t>);
  void mark_split_in_progress(spg_t, const set<spg_t>);
  void cancel_pending_splits_for_parent(spg_t);
  void _cancel_pending_splits_for_parent(spg_t parent);
  bool splitting(spg_t);
  void expand_pg_num(OSDMapRef, OSDMapRef);
  void _maybe_split_pgid(OSDMapRef, OSDMapRef, spg_t);
  void init_splits_between(spg_t, OSDMapRef, OSDMapRef);
  Mutex stat_lock;
  osd_stat_t osd_stat;
  uint32_t seq;
  void update_osd_stat(vector<int>&);
  osd_stat_t set_osd_stat(const struct store_statfs_t, vector<int>&, int);
  osd_stat_t get_osd_stat();
  uint64_t get_osd_stat_seq();

  bool check_failsafe_full(ostream&) const;
  bool check_full(ostream&) const;
  bool check_backfill_full(ostream&) const;
  bool check_nearfull(ostream&) const;
  bool is_failsafe_full() const;
  bool is_full() const;
  bool is_backfillfull() const;
  bool is_neaffull() const;
  // osdmap state needs update
  bool need_fullness_update();
  void set_injectfull(s_names, int64_t);
  bool check_osdmap_full(const set<pg_shared_t>&);

  //  Retrieve the boot_, up_, and bind_ epochs the OSD has set. The params
  // can be NULL if you don't care about them.
  void retrieve_epochs(epoch_t*, epoch_t*, epoch_t*) const;
  // Set the boot, up, and bind epochs. Any NULL params will not be set.
  void set_epochs(const epoch_t*, const epoch_t*, const epoch_t*);
  epoch_t get_boot_epoch() const;
  epoch_t get_up_epoch() const;
  epoch_t get_bind_epoch() const;
  void request_osdmap_update(epoch_t);
  Mutex is_stopping_lock;
  Cond is_stopping_cond;
  enum {
    NOT_STOPPING, PREPARING_TO_STOP, STOPPING };
  std::atomic_int state{NOT_STOPPING};
  int get_state() const;
  void set_state(int);
  bool is_stopping() const;
  bool is_preparing_to_stop() const;
  bool prepare_to_stop();
  void got_stop_ack();

private:
  Messenger*& cluster_messenger;
  Messenger*& client_messenger;

  // map epoch lower bound
  Mutex pg_epoch_lock;
  Cond pg_cond;
  multiset<epoch_t> pg_epochs;
  map<spg_t, epoch_t> pg_epoch;

  // pre-publish orders before publish
  Mutex publish_lock, pre_publish_lock;
  OSDSuperblock superblock;

  OSDMapRef osdmap;

  OSDMapRef next_osdmap;
  Cond pre_pubish_cond;

  Mutex peer_map_epoch_lock;
  map<int, epoch_t> peer_map_epoch;

  Mutex sched_scrub_lock;
  int scrubs_pending;
  int scrubs_active;

  Mutex agent_lock;
  Cond agent_cond;
  map<uint64_t, set<PGRef> > agent_queue;
  set<PGRef>::iterator agent_queue_pos;
  bool agent_valid_iterator;
  int agent_ops;
  int flush_mode_high_count; // ?
  // once have one pg with FLUSH_MODE_HIGH then flush objects with high
  // speed
  set<hobject_t> agent_oids;
  bool agent_active;

  /* AgentThread */
  AgentThread : Thread
    OSDService* osd;
    explicit AgentThread(OSDService*);
    void* entry() override;
  /* AgentThread */

  bool agent_stop_flag;
  Mutex agent_timer_lock;
  SafeTimer agent_timer;

  // throttle promotion attempts
  std::atomic<unsigned int> promote_probability_millis{1000};
  PromoteCounter promote_counter;
  utime_t last_recalibrate;
  unsigned long promote_max_objects, promote_max_bytes;

  Mutex pg_temp_lock;
  map<pg_t, vector<int> > pg_temp_wanted;
  map<pg_t, vector<int> > pg_temp_pending;
  void _sent_pg_temp();

  Mutex recovery_lock;
  list<pair<epoch_t, PGRef> > awaiting_throttle;
  utime_t defer_recovery_until;
  uint64_t recovery_ops_active;
  uint64_t recovery_ops_reserved;
  bool recovery_paused;
  map<spg_t, set<hobject_t> > recovery_oids;
  bool _recover_now(uint64_t*);
  void _maybe_queue_recovery();
  void _queue_for_recovery(pair<epoch_t, PGRef>, uint64_t);

  Mutex in_progress_split_lock;
  // child -> parent
  map<spg_t, spg_t> pending_splits;
  // parent -> children
  map<spg_t, set<spg_t> > rev_pending_splits;
  // child
  set<spg_t> in_progress_splists;

  friend TestOpsSocketHook;
  mutable Mutex full_status_lock;
  enum s_names {
    INVALID, NONE, NEARFULL, BACKFILL, FULL, FAILSAFE } cur_state;
  const char* get_full_state_name(s_names) const;
  s_names get_full_state(string) const;
  // current utilizatio
  double cur_ratio;
  mutable int64_t injectfull = 0;
  s_names injectfull_state = NONE;
  float get_failsafe_full_ratio();
  void check_full_status(float);
  bool _check_full(s_names, ostream&) const;

  // utex epoch_lock; // protects access to boot_epoch, up_epoch,
  // bind_epoch
  mutable Mutex epoch_lock;
  // _first_ epoch we were marked up (after this process started)
  epoch_t boot_epoch;
  // _most_recent_ epoch we were marked up
  epoch_t up_epoch;
  // epoch we last did a bind to new ip:ports
  epoch_t bind_epoch

// @OSDService

class OSD : public Dispatcher, public md_config_obs_t

