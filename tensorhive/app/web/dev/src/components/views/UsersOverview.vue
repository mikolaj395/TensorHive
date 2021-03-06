<template>
  <section class="content">
    <v-alert
      v-model="alert"
      dismissible
      type="error"
    >
      {{ errorMessage }}
    </v-alert>
    <v-alert
      v-model="created"
      dismissible
      type="info"
    >
      User successfully created
    </v-alert>
    <v-layout row justify-center>
      <v-dialog
        width="500px"
        v-model="showModal"
      >
        <v-card>
          <v-card-text>
            <v-btn
              class="float-right-button"
              flat
              icon
              color="black"
              @click="showModal=false"
            >
              <v-icon>close</v-icon>
            </v-btn>
            <span class="headline">Create new user</span>
          </v-card-text>
          <v-card-text>
            <form @submit.prevent="createUser">
              <v-card-text>
                Username
              </v-card-text>
              <div class="input-group">
                <span class="input-group-addon"><i class="fa fa-user"></i></span>
                <input
                  class="form-control"
                  name="modalUsername"
                  placeholder="Username"
                  type="text"
                  v-model="modalUsername"
                >
              </div>
              Email
              <div class="input-group">
                <span class="input-group-addon"><i class="fa fa-envelope"></i></span>
                <input
                  class="form-control"
                  name="modalEmail"
                  placeholder="Email"
                  type="text"
                  v-model="modalEmail"
                >
              </div>
              Password
              <div class="input-group">
                <span class="input-group-addon"><i class="fa fa-lock"></i></span>
                <input
                  class="form-control"
                  name="modalPassword"
                  placeholder="Password"
                  type="password"
                  v-model="modalPassword"
                >
              </div>
              Repeat password
              <div class="input-group">
                <span class="input-group-addon"><i class="fa fa-lock"></i></span>
                <input
                  class="form-control"
                  name="modalPassword2"
                  placeholder="Password2"
                  type="password"
                  v-model="modalPassword2"
                >
              </div>
              <v-alert
                v-model="modalAlert"
                dismissible
                type="error"
              >
                {{ errorMessage }}
              </v-alert>
              <v-btn
                color="success"
                type="submit"
              >
                Create
              </v-btn>
            </form>
          </v-card-text>
        </v-card>
      </v-dialog>
    </v-layout>
    <v-dialog
      v-model="showModalRemove"
      width="400"
    >
      <v-card>
        <v-card-text
          class="headline grey lighten-2"
          primary-title
        >
          <v-btn
            class="float-right-button"
            flat
            icon
            color="black"
            @click="showModalRemove= false"
          >
            <v-icon>close</v-icon>
          </v-btn>
          Do you want to remove this user?
        </v-card-text>
        <v-card-actions>
          <v-layout align-center justify-end>
            <v-btn
              color="success"
              round
              @click="removeUser()"
            >
              Yes
            </v-btn>
          </v-layout>
        </v-card-actions>
      </v-card>
    </v-dialog>
    <div>
      <div class="text-xs-center pt-2">
        <v-btn color="primary" @click="showModal=true">Create user</v-btn>
      </div>
      <v-dialog v-model="dialog" max-width="500px">
        <v-card>
          <v-card-text>
            <v-btn
              class="float-right-button"
              flat
              icon
              color="black"
              @click="dialog = false"
            >
              <v-icon>close</v-icon>
            </v-btn>
            <v-card-text>
              Edit user
            </v-card-text>
            <v-card-text>
              Current username: {{currentUser.username}}
            </v-card-text>
            <v-card-text>
              New username
            </v-card-text>
            <div class="input-group">
              <span class="input-group-addon"><i class="fa fa-envelope"></i></span>
              <input
                class="form-control"
                name="modalUsername"
                placeholder="Username"
                type="text"
                v-model="user.username"
              >
            </div>
            <v-card-text>
              Current email: {{currentUser.email}}
            </v-card-text>
            <v-card-text>
              New email
            </v-card-text>
            <div class="input-group">
              <span class="input-group-addon"><i class="fa fa-envelope"></i></span>
              <input
                class="form-control"
                name="modalEmail"
                placeholder="Email"
                type="text"
                v-model="user.email"
              >
            </div>
            <v-card-text>
              New password
            </v-card-text>
            <div class="input-group">
              <span class="input-group-addon"><i class="fa fa-lock"></i></span>
              <input
                class="form-control"
                name="modalPassword"
                placeholder="Password"
                type="password"
                v-model="user.password"
              >
            </div>
            <v-card-text>
              Repeat password
            </v-card-text>
            <div class="input-group">
              <span class="input-group-addon"><i class="fa fa-lock"></i></span>
              <input
                class="form-control"
                name="modalPassword2"
                placeholder="Password2"
                type="password"
                v-model="user.password2"
              >
            </div>
            <v-card-text>
              Account roles:
            </v-card-text>
            <v-card-text>
              <v-checkbox
                label="admin"
                v-model="adminCheckbox"
              >
              </v-checkbox>
            </v-card-text>
          </v-card-text>
          <v-card-actions>
            <v-spacer></v-spacer>
            <v-btn color="blue darken-1" flat @click="updateUser">Edit</v-btn>
          </v-card-actions>
        </v-card>
      </v-dialog>
      <v-data-table
        :headers="headers"
        :items="users"
        :search="search"
        :pagination.sync="pagination"
        item-key="id"
        hide-actions
        class="elevation-1"
      >
        <template slot="items" slot-scope="props">
          <tr>
            <td>{{ props.item.id }}</td>
            <td>{{ props.item.username }}</td>
            <td>{{ props.item.email }}</td>
            <td>{{ prettyDate(props.item.createdAt) }}</td>
            <td>{{ props.item.role }}</td>
            <td>
              <v-icon
                small
                @click="editUser(props.item)"
              >
                edit
              </v-icon>
              <v-icon
                small
                @click="showConfirmationDialog(props.item.id)"
              >
                delete
              </v-icon>
            </td>
          </tr>
        </template>
      </v-data-table>
      <div class="text-xs-center pt-2">
        <v-pagination v-model="pagination.page" :length="pages"></v-pagination>
      </div>
    </div>
  </section>
</template>

<script>
import api from '../../api'
import moment from 'moment'
export default {
  data () {
    return {
      dialog: false,
      search: '',
      pagination: {},
      selected: [],
      headers: [
        { text: 'User id', value: 'id' },
        { text: 'Username', value: 'username' },
        { text: 'Email', value: 'email' },
        { text: 'Created at', value: 'createdAt' },
        { text: 'Role', value: 'role' },
        { text: 'Actions', value: 'id' }
      ],
      users: [],
      user: {
        id: -1,
        username: '',
        email: '',
        password: '',
        password2: '',
        roles: []
      },
      currentUser: {},
      time: 1000,
      alert: false,
      errorMessage: '',
      userCheckbox: false,
      adminCheckbox: false,
      modalUsername: '',
      modalEmail: '',
      modalPassword: '',
      modalPassword2: '',
      modalAlert: false,
      showModalRemove: false,
      userId: -1,
      created: false,
      showModal: false
    }
  },

  computed: {
    pages () {
      if (this.pagination.rowsPerPage == null ||
        this.pagination.totalItems == null
      ) return 0

      return Math.ceil(this.pagination.totalItems / this.pagination.rowsPerPage)
    }
  },

  mounted () {
    this.checkUsers()
  },

  methods: {
    prettyDate (date) {
      if (date !== null) {
        return moment(date).format('dddd, MMMM Do, HH:mm')
      } else {
        return null
      }
    },

    handleError: function (error) {
      if (!error.hasOwnProperty('response')) {
        this.errorMessage = error.message
      } else {
        if (!error.response.data.hasOwnProperty('msg')) {
          this.errorMessage = error.response.data
        } else {
          this.errorMessage = error.response.data.msg
        }
      }
    },

    createUser () {
      if (this.modalPassword === this.modalPassword2) {
        const { modalUsername, modalEmail, modalPassword } = this
        api
          .request('post', '/user/create', this.$store.state.accessToken, { 'username': modalUsername, 'email': modalEmail, 'password': modalPassword })
          .then(response => {
            this.showModal = false
            this.created = true
            this.checkUsers()
          })
          .catch(error => {
            this.handleError(error)
            this.modalAlert = true
          })
      } else {
        this.errorMessage = 'Passwords do not match'
        this.modalAlert = true
      }
    },

    editUser: function (currentUser) {
      this.dialog = true
      this.user.id = currentUser.id
      this.user.username = currentUser.username
      this.user.email = currentUser.email
      var admin = false
      for (var role in currentUser.roles) {
        if (currentUser.roles[role] === 'admin') {
          admin = true
        }
      }
      this.adminCheckbox = admin
      this.currentUser = currentUser
    },

    updateUser: function () {
      if (this.user.password === this.user.password2) {
        if (this.adminCheckbox) {
          this.user.roles.push('admin')
        }
        this.user.roles.push('user')
        var updatedUser = {
          id: this.user.id
        }
        if (this.user.username !== this.currentUser.username && this.user.username !== '') {
          updatedUser['username'] = this.user.username
        }
        if (this.user.email !== this.currentUser.email && this.user.email !== '') {
          updatedUser['email'] = this.user.email
        }
        if (this.user.password !== '') {
          updatedUser['password'] = this.user.password
        }
        if (this.user.roles.length !== this.currentUser.roles.length) {
          updatedUser['roles'] = this.user.roles
        }
        api
          .request('put', '/user', this.$store.state.accessToken, updatedUser)
          .then(response => {
            this.user = {
              id: -1,
              username: '',
              email: '',
              password: '',
              password2: '',
              roles: []
            }
            this.adminCheckbox = false
            this.userCheckbox = false
            this.dialog = false
            this.checkUsers()
          })
          .catch(error => {
            this.pagination = {}
            this.handleError(error)
            this.alert = true
          })
      } else {
        this.errorMessage = 'Passwords do not match'
        this.alert = true
      }
    },
    checkUsers: function () {
      api
        .request('get', '/users', this.$store.state.accessToken)
        .then(response => {
          this.users = response.data
          for (var user in this.users) {
            var admin = false
            for (var role in this.users[user].roles) {
              if (this.users[user].roles[role] === 'admin') {
                admin = true
              }
            }
            if (admin) {
              this.users[user]['role'] = 'admin'
            } else {
              this.users[user]['role'] = 'user'
            }
          }
          this.pagination['totalItems'] = this.users.length
          this.pagination['rowsPerPage'] = 30
        })
        .catch(error => {
          this.pagination = {}
          this.handleError(error)
          this.alert = true
        })
    },

    showConfirmationDialog (id) {
      this.userId = id
      this.showModalRemove = true
    },

    removeUser: function () {
      var userId = this.userId
      api
        .request('delete', '/user/delete/' + userId, this.$store.state.accessToken)
        .then(response => {
          this.showModalRemove = false
          this.checkUsers()
        })
        .catch(error => {
          this.handleError(error)
          this.alert = true
        })
    }
  }
}
</script>

<style scoped>
.float-right-button {
  float: right;
}
.input-group {
  padding-bottom: 2em;
  height: 4em;
  width: 100%;
}

.input-group span.input-group-addon {
  width: 2em;
  height: 4em;
}

@media (max-width: 1241px) {
  .input-group input {
    height: 4em;
  }
}
@media (min-width: 1242px) {
  .input-group input {
    height: 6em;
  }
}

.input-group-addon i {
  height: 15px;
  width: 15px;
}
</style>
