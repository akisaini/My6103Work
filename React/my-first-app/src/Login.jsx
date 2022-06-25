import { Component } from "react";


export default class Login extends Component {

constructor(props){
super(props);
this.state = {
    email:'',
    password:'',
    message:'', 
}
}


render() {

    return <div className='col-lg-9'>
        <h4 className='m-1 p-2 border-bo6ttom'>Login</h4>

{/*Email form row Starts*/}
        <div className='form-group form-row'>
            <label className='col-lg-4'>Email:</label>
            <input type = 'text' className='form-control' value = {this.state.email} onChange = {(event) => {this.setState({email:event.target.value})}}></input>
            {/*The event method allows us to access the latest value in the browser Component that was updated. */}
        </div>
{/*Email Ends*/}

{/*Password form row Starts*/}
        <div className='form-group form-row'>
            <label className='col-lg-4'>Password:</label>
            <input type = 'password' className='form-control' value = {this.state.password} onChange = {(event) => {this.setState({password:event.target.value})}}></input>
        </div>
{/*Password Ends*/}
        <div className='text-right'>
        {this.state.message}
            <button className='btn btn-primary m-1' onClick ={ () => {
                        this.onLoginClick()}}>Login</button>
        </div>

    </div>


}

onLoginClick = () => {
if (this.state.email === 'admin@test.com' && this.state.password === 'admin123'){
    //success
    this.setState({message:<span className='text-success'>Successfully Logged-in</span>})
}
else {
    this.setState({message:<span className='text-danger'>Invalid Login, please try again</span>})
}
}

} // Component Parenthesis
